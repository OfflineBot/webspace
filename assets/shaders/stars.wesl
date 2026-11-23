import bevy_pbr::render::forward_io::VertexOutput;

// ---------------------------------------------------------------- stars

/// one layer per size class. SIZE * HALO_REACH * DENSITY is the margin a star
/// needs inside its cell. keep it well under 0.5: at 0.45 the star is pinned
/// to the cell centre and the grid becomes visible as rows
const STAR_DENSITY_FAR: f32 = 40.0;
const STAR_DENSITY_MID: f32 = 18.0;
const STAR_DENSITY_NEAR: f32 = 8.0;

/// fraction of cells that hold a star at all
const STAR_CHANCE_FAR: f32 = 0.70;
const STAR_CHANCE_MID: f32 = 0.35;
const STAR_CHANCE_NEAR: f32 = 0.06;

/// radius of the brightest star of a layer, in uv units
const STAR_SIZE_FAR: f32 = 0.0016;
const STAR_SIZE_MID: f32 = 0.0035;
const STAR_SIZE_NEAR: f32 = 0.0070;

/// under roughly 1.5 pixels a star flickers while turning, never go below
const STAR_SIZE_MIN: f32 = 0.0014;

/// how far the glow reaches, as a multiple of the core radius
const STAR_HALO_REACH: f32 = 2.0;
const STAR_HALO_STRENGTH: f32 = 0.10;

/// falloff sharpness of core and halo. higher is a tighter point.
/// an exponential never reaches zero, so HALO_FALLOFF also decides whether
/// the glow is already invisible at the cell border or gets clipped there
const STAR_CORE_FALLOFF: f32 = 4.0;
const STAR_HALO_FALLOFF: f32 = 4.0;

/// safety factor on the margin, for the bit of glow beyond HALO_REACH
const STAR_CLIP_GUARD: f32 = 1.3;

/// output scale, and how steeply star counts grow towards faint
const STAR_BRIGHTNESS: f32 = 1.0;
const STAR_FAINT_BIAS: f32 = 3.5;

/// blackbody-ish colours, cool to hot
const STAR_COOL: vec3<f32> = vec3<f32>(1.00, 0.48, 0.26);
const STAR_WARM: vec3<f32> = vec3<f32>(1.00, 0.82, 0.62);
const STAR_NEUTRAL: vec3<f32> = vec3<f32>(1.00, 0.96, 0.92);
const STAR_HOT: vec3<f32> = vec3<f32>(0.58, 0.72, 1.00);

/// the eye reads colour only in bright stars, so faint ones stay grey
const STAR_SATURATION_MIN: f32 = 0.15;
const STAR_SATURATION_MAX: f32 = 1.00;

// --------------------------------------------------------------- galaxy

/// normal of the galactic plane
const GALAXY_AXIS: vec3<f32> = vec3<f32>(0.19, 0.96, 0.21);

/// higher is a tighter band
const GALAXY_WIDTH: f32 = 4.0;

const GALAXY_BRIGHTNESS: f32 = 0.20;

/// the band alone is a closed ring. this large scale field decides where
/// there is any cloud at all, so the ring breaks up into separate masses
const GALAXY_MASS_SCALE: f32 = 1.7;
const GALAXY_MASS_GATE_LOW: f32 = 0.36;
const GALAXY_MASS_GATE_HIGH: f32 = 0.70;

/// size of the cloud detail, and of the slower hue variation on top of it
const GALAXY_CLOUD_SCALE: f32 = 6.0;
const GALAXY_HUE_SCALE: f32 = 2.2;

/// higher means darker gaps between the clouds
const GALAXY_DUST_CONTRAST: f32 = 2.2;

/// fbm clusters around the middle, so without spreading the hue every cloud
/// comes out violet. this pushes it to the ends of the palette
const GALAXY_HUE_SPREAD_LOW: f32 = 0.32;
const GALAXY_HUE_SPREAD_HIGH: f32 = 0.68;

/// the red to blue ramps overlap, which removes the kink a hard split at 0.5
/// leaves in the gradient
const GALAXY_HUE_RED_END: f32 = 0.62;
const GALAXY_HUE_BLUE_START: f32 = 0.38;

/// full colour only in the dense parts. everywhere else fades towards this,
/// so the sky is not uniformly saturated
const GALAXY_FAINT: vec3<f32> = vec3<f32>(0.18, 0.20, 0.34);
const GALAXY_SATURATE_LOW: f32 = 0.10;
const GALAXY_SATURATE_HIGH: f32 = 0.55;

const GALAXY_RED: vec3<f32> = vec3<f32>(0.70, 0.11, 0.16);
const GALAXY_VIOLET: vec3<f32> = vec3<f32>(0.42, 0.14, 0.70);
const GALAXY_BLUE: vec3<f32> = vec3<f32>(0.05, 0.18, 0.80);

/// the dense parts wash out towards this
const GALAXY_CORE: vec3<f32> = vec3<f32>(0.72, 0.68, 0.92);
const GALAXY_CORE_REACH: f32 = 2.2;

/// how much more crowded the band is than empty sky
const GALAXY_STAR_BOOST: f32 = 1.7;

// ----------------------------------------------------------------- hash

/// pcg, from Jarzynski and Olano, "Hash Functions for GPU Rendering".
/// integer hashing has no precision ceiling, unlike the usual sin() trick
fn pcg4d(v_in: vec4<u32>) -> vec4<u32> {
    var v = v_in * 1664525u + 1013904223u;
    v.x += v.y * v.w;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.w += v.y * v.z;
    v = v ^ (v >> vec4<u32>(16u));
    v.x += v.y * v.w;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.w += v.y * v.z;
    return v;
}

/// casting a negative float straight to u32 saturates to zero, which would
/// collapse every negative cell onto one hash. bitcast keeps the bits
fn hash4(p: vec4<f32>) -> vec4<f32> {
    let h = pcg4d(bitcast<vec4<u32>>(vec4<i32>(p)));
    return vec4<f32>(h) / 4294967295.0;
}

fn hash1(p: vec3<f32>) -> f32 {
    return hash4(vec4<f32>(p, 0.0)).x;
}

// ---------------------------------------------------------------- noise

fn value_noise(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);

    let c000 = hash1(i + vec3<f32>(0.0, 0.0, 0.0));
    let c100 = hash1(i + vec3<f32>(1.0, 0.0, 0.0));
    let c010 = hash1(i + vec3<f32>(0.0, 1.0, 0.0));
    let c110 = hash1(i + vec3<f32>(1.0, 1.0, 0.0));
    let c001 = hash1(i + vec3<f32>(0.0, 0.0, 1.0));
    let c101 = hash1(i + vec3<f32>(1.0, 0.0, 1.0));
    let c011 = hash1(i + vec3<f32>(0.0, 1.0, 1.0));
    let c111 = hash1(i + vec3<f32>(1.0, 1.0, 1.0));

    let x00 = mix(c000, c100, u.x);
    let x10 = mix(c010, c110, u.x);
    let x01 = mix(c001, c101, u.x);
    let x11 = mix(c011, c111, u.x);

    return mix(mix(x00, x10, u.y), mix(x01, x11, u.y), u.z);
}

fn fbm(p: vec3<f32>) -> f32 {
    var sum = 0.0;
    var amplitude = 0.5;
    var frequency = 1.0;

    for (var i = 0; i < 3; i++) {
        sum += value_noise(p * frequency) * amplitude;
        frequency *= 2.17;
        amplitude *= 0.5;
    }

    return sum / 0.875;
}

// ------------------------------------------------------------ cube face

/// a 3d grid on a sphere direction gives wildly uneven cells, because the
/// sphere cuts some cubes through the middle and others at a corner. project
/// onto the cube face instead and work in 2d, where every cell is a square.
///
/// returns (u, v) in -1..1 plus the face index, which has to go into the hash
/// or opposite faces share their stars
fn cube_face(dir: vec3<f32>) -> vec3<f32> {
    let a = abs(dir);

    if a.x >= a.y && a.x >= a.z {
        let face = select(1.0, 0.0, dir.x > 0.0);
        return vec3<f32>(dir.y / a.x, dir.z / a.x, face);
    }
    if a.y >= a.z {
        let face = select(3.0, 2.0, dir.y > 0.0);
        return vec3<f32>(dir.x / a.y, dir.z / a.y, face);
    }
    let face = select(5.0, 4.0, dir.z > 0.0);
    return vec3<f32>(dir.x / a.z, dir.y / a.z, face);
}

/// the projection above packs cells tighter towards the face edges, which
/// reads as a visible cube. this spreads them evenly again
fn equalise(v: vec2<f32>) -> vec2<f32> {
    return atan(v) * 1.2732395;
}

// ----------------------------------------------------------------- star

fn star_colour(t: f32) -> vec3<f32> {
    if t < 0.33 {
        return mix(STAR_COOL, STAR_WARM, t * 3.0);
    }
    if t < 0.66 {
        return mix(STAR_WARM, STAR_NEUTRAL, (t - 0.33) * 3.0);
    }
    return mix(STAR_NEUTRAL, STAR_HOT, (t - 0.66) * 2.94);
}

fn star_layer(
    uv_base: vec2<f32>,
    face: f32,
    density: f32,
    chance: f32,
    size_max: f32,
    crowding: f32,
    seed: f32,
) -> vec3<f32> {
    // the seed offsets the grid so the layers do not line up on each other
    let uv = (uv_base + vec2<f32>(seed)) * density;
    let cell = floor(uv);
    let h = hash4(vec4<f32>(cell, face, seed));

    let threshold = chance * crowding;
    if h.z > threshold {
        return vec3<f32>(0.0);
    }

    // reuse the existence value as the magnitude, then bias it hard towards
    // faint: real star counts grow steeply as brightness drops
    let mag = pow(h.z / threshold, STAR_FAINT_BIAS);

    // a star is a point source, its apparent size comes from brightness alone
    let size = mix(STAR_SIZE_MIN, max(STAR_SIZE_MIN, size_max), mag);
    let reach = size * STAR_HALO_REACH;

    // hold the star far enough inside its cell that core and halo both fit
    let margin = min(0.45, reach * STAR_CLIP_GUARD * density);
    let pos = cell + vec2<f32>(margin) + h.xy * (1.0 - 2.0 * margin);

    // exponential falloff, not smoothstep: a smoothstep across the whole
    // radius is a soft blob, a star is a tight point with a faint skirt
    let d = distance(uv, pos) / density;
    let core = exp(-pow(d / size, 2.0) * STAR_CORE_FALLOFF);
    let halo = exp(-pow(d / reach, 2.0) * STAR_HALO_FALLOFF) * STAR_HALO_STRENGTH;

    let saturation = mix(STAR_SATURATION_MIN, STAR_SATURATION_MAX, mag);
    let tint = mix(vec3<f32>(1.0), star_colour(h.w), saturation);

    return tint * (core + halo) * mag * STAR_BRIGHTNESS;
}

// ------------------------------------------------------------- fragment

/// 0.0 is the far plane in reverse z and the clear value of the depth buffer,
/// so the sky can never cover geometry that is farther out than the sphere
struct StarOutput {
    @builtin(frag_depth) depth: f32,
    @location(0) color: vec4<f32>,
}

@fragment
fn fragment(in: VertexOutput) -> StarOutput {
    let dir = normalize(in.world_normal);

    // ---- galaxy

    // 1 along the galactic plane, 0 at its poles
    let plane = 1.0 - abs(dot(dir, normalize(GALAXY_AXIS)));
    let band = pow(plane, GALAXY_WIDTH);

    // where there is cloud at all. gating a large scale field breaks the band
    // open, so it reads as separate masses instead of one closed ring
    let mass_raw = fbm(dir * GALAXY_MASS_SCALE + vec3<f32>(5.0, 91.0, 17.0));
    let mass = smoothstep(GALAXY_MASS_GATE_LOW, GALAXY_MASS_GATE_HIGH, mass_raw);

    // detail inside the masses
    let clouds = pow(fbm(dir * GALAXY_CLOUD_SCALE), GALAXY_DUST_CONTRAST);
    let intensity = band * mass * clouds;

    // hue varies on its own, slower scale, so colour is not just a function
    // of brightness
    let hue_raw = fbm(dir * GALAXY_HUE_SCALE + vec3<f32>(37.0, 11.0, 73.0));
    let hue = smoothstep(GALAXY_HUE_SPREAD_LOW, GALAXY_HUE_SPREAD_HIGH, hue_raw);

    // two overlapping ramps instead of a hard split at 0.5, which would leave
    // a visible kink where the gradient changes direction
    let warm = mix(GALAXY_RED, GALAXY_VIOLET, smoothstep(0.0, GALAXY_HUE_RED_END, hue));
    let nebula = mix(warm, GALAXY_BLUE, smoothstep(GALAXY_HUE_BLUE_START, 1.0, hue));

    // only the dense parts carry full colour
    let saturation = smoothstep(GALAXY_SATURATE_LOW, GALAXY_SATURATE_HIGH, intensity);
    let tinted = mix(GALAXY_FAINT, nebula, saturation);

    let core = pow(intensity, GALAXY_CORE_REACH);
    var colour = mix(tinted, GALAXY_CORE, core) * intensity * GALAXY_BRIGHTNESS;

    // ---- stars

    let f = cube_face(dir);
    let uv = equalise(f.xy);
    let face = f.z;

    // the band is more crowded than empty sky
    let crowding = mix(1.0, GALAXY_STAR_BOOST, intensity);

    colour += star_layer(uv, face, STAR_DENSITY_FAR, STAR_CHANCE_FAR,
        STAR_SIZE_FAR, crowding, 0.0);
    colour += star_layer(uv, face, STAR_DENSITY_MID, STAR_CHANCE_MID,
        STAR_SIZE_MID, crowding, 11.0);
    colour += star_layer(uv, face, STAR_DENSITY_NEAR, STAR_CHANCE_NEAR,
        STAR_SIZE_NEAR, crowding, 23.0);

    return StarOutput(0.0, vec4<f32>(colour, 1.0));
}
