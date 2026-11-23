import bevy_pbr::render::forward_io::VertexOutput;
import bevy_pbr::render::mesh_view_bindings::{view, lights};
import bevy_pbr::render::mesh_functions;
import bevy_pbr::prepass::utils::prepass_depth;

/// all lengths are world units. scattering and falloff are converted from the
/// authored meters on the cpu side, so this shader never sees two unit systems
struct Air {
    scattering: vec3<f32>,
    inner: f32,
    outer: f32,
    falloff: f32,
}

@group(constants::MATERIAL_BIND_GROUP) @binding(0) var<uniform> air: Air;

const STEPS: i32 = 16;
const SUN_STEPS: i32 = 4;

/// 3 / (16 * PI), the normalisation of the rayleigh phase function
const RAYLEIGH: f32 = 0.0596831;

/// overall brightness of the scattered light
const STRENGTH: f32 = 1.4;

/// how bright air in the planet's shadow still is, as a fraction of lit air.
/// single scattering puts nothing there at all, and a shell that simply ends at
/// the terminator reads as a hole
const NIGHT: f32 = 0.012;

/// distances along the ray to a sphere of radius r centred on the origin.
/// x > y means the ray misses it.
///
/// solved through the closest approach rather than the usual discriminant:
/// b * b and dot(o, o) cancel almost completely when the camera is far out,
/// which eats most of the f32 mantissa
fn hit(o: vec3<f32>, d: vec3<f32>, r: f32) -> vec2<f32> {
    let m = -dot(o, d);
    let q = o + d * m;
    let h2 = r * r - dot(q, q);
    if h2 < 0.0 {
        return vec2<f32>(1.0, -1.0);
    }
    let h = sqrt(h2);
    return vec2<f32>(m - h, m + h);
}

fn density(p: vec3<f32>) -> f32 {
    return exp(-max(length(p) - air.inner, 0.0) / air.falloff);
}

/// how much of the sun reaches p. the ray towards the sun passes the planet
/// centre at some distance, and below the surface that is shadow. the scale
/// height as the width makes the terminator a soft band instead of an edge
fn sun_shade(p: vec3<f32>, sun: vec3<f32>) -> f32 {
    let m = -dot(p, sun);
    if m <= 0.0 {
        return 1.0;
    }
    let grazing = length(p + sun * m);
    return smoothstep(air.inner - air.falloff, air.inner + air.falloff, grazing);
}

/// maps u in 0..1 to a distance along the ray, with the samples packed
/// quadratically around low. air thins exponentially, so evenly spaced samples
/// spend nearly all of them in vacuum and miss the dense layer entirely
fn place(u: f32, start: f32, end: f32, low: f32) -> f32 {
    let before = low - start;
    let after = end - low;
    let span = before + after;
    if span <= 0.0 {
        return start;
    }
    let split = before / span;
    if u < split {
        let s = 1.0 - u / max(split, 1e-6);
        return low - before * s * s;
    }
    let s = (u - split) / max(1.0 - split, 1e-6);
    return low + after * s * s;
}

/// density integrated from p towards the sun
fn to_sun(p: vec3<f32>, sun: vec3<f32>) -> f32 {
    let shell = hit(p, sun, air.outer);
    let far = max(shell.y, 0.0);

    var sum = 0.0;
    var prev = 0.0;
    for (var i = 0; i < SUN_STEPS; i++) {
        let u = f32(i + 1) / f32(SUN_STEPS);
        let next = far * u * u;
        sum += density(p + sun * ((prev + next) * 0.5)) * (next - prev);
        prev = next;
    }
    return sum;
}

/// the star is the brightest directional light. the others are fill light for
/// geometry, and the index in the light array is not ordered
fn star_index() -> u32 {
    var best = 0u;
    var best_power = -1.0;
    for (var l = 0u; l < lights.n_directional_lights; l++) {
        let power = dot(lights.directional_lights[l].color.rgb, vec3<f32>(1.0));
        if power > best_power {
            best_power = power;
            best = l;
        }
    }
    return best;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // the centre comes from the model matrix, not from position minus normal:
    // the shell is a polyhedron, so those two only agree at its vertices and the
    // reconstructed centre wanders by the facet sag in between, which the
    // exponential density turns into visible patches on the ground
    let centre = (mesh_functions::get_world_from_local(in.instance_index) * vec4<f32>(0.0, 0.0, 0.0, 1.0)).xyz;

    let origin = view.world_position - centre;
    let dir = normalize(in.world_position.xyz - view.world_position);

    let shell = hit(origin, dir, air.outer);
    if shell.y < shell.x || shell.y <= 0.0 {
        discard;
    }

    let start = max(shell.x, 0.0);
    var end = shell.y;

    let ground = hit(origin, dir, air.inner);
    if ground.y > ground.x && ground.x > start {
        end = min(end, ground.x);
    }
    // the shell ignores the depth test so that it also covers the planet disc,
    // which means the ray has to be stopped at the scene depth by hand
    let scene = prepass_depth(in.position, 0u);
    if scene > 0.0 {
        let ndc = vec3<f32>(
            in.position.x / view.viewport.z * 2.0 - 1.0,
            1.0 - in.position.y / view.viewport.w * 2.0,
            scene,
        );
        let world = view.world_from_clip * vec4<f32>(ndc, 1.0);
        end = min(end, length(world.xyz / world.w - view.world_position));
    }

    if end <= start {
        discard;
    }

    let star = star_index();

    // the lowest point of the ray carries the densest air. clamped, it is the
    // camera itself for anything looking up from the ground
    let low = clamp(-dot(origin, dir), start, end);

    // a fixed step count over a smoothly changing ray length puts the brightness
    // steps on lines of equal distance, which read as rings. offsetting the first
    // sample per pixel turns that into noise the eye does not resolve
    let jitter = fract(sin(dot(in.position.xy, vec2<f32>(12.9898, 78.233))) * 43758.5453);
    var depth = 0.0;
    var lit = 0.0;
    var scattered = vec3<f32>(0.0);
    var prev = start;

    for (var i = 0; i < STEPS; i++) {
        let next = place(f32(i + 1) / f32(STEPS), start, end, low);
        let step = next - prev;
        let p = origin + dir * mix(prev, next, jitter);
        prev = next;
        let mass = density(p) * step;
        depth += mass;

        let towards_view = exp(-air.scattering * depth);

        if lights.n_directional_lights > 0u {
            let sun = lights.directional_lights[star].direction_to_light;
            let shade = sun_shade(p, sun);

            // what the sun still delivers here, plus the floor for shadowed air
            let reach = exp(-air.scattering * to_sun(p, sun)) * shade
                + NIGHT * (1.0 - shade);

            lit += mass * (shade + NIGHT * (1.0 - shade));

            let mu = dot(dir, sun);
            let phase = RAYLEIGH * (1.0 + mu * mu);

            scattered += lights.directional_lights[star].color.rgb
                * phase
                * air.scattering
                * mass
                * towards_view
                * reach;
        }
    }

    // only air that sees the sun hides what is behind it. extinction alone would
    // put a black veil over the night side, where single scattering has nothing
    // to put back in
    let opacity = 1.0 - exp(-dot(air.scattering, vec3<f32>(0.3333333)) * lit);
    return vec4<f32>(scattered * STRENGTH * view.exposure, opacity);
}
