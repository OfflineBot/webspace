import bevy_pbr::render::forward_io::VertexOutput;
import bevy_pbr::render::mesh_view_bindings::view;

@group(constants::MATERIAL_BIND_GROUP) @binding(0) var<uniform> settings: vec4<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(1) var<uniform> falloff: vec4<f32>;

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let n = normalize(in.world_normal);
    let v = normalize(view.world_position.xyz - in.world_position.xyz);
    let f = falloff.x + falloff.y * pow(1.0 - abs(dot(n, v)), settings.w);
    return vec4(settings.xyz * f, 0.0);
}
