// 最简单的测试shader，模仿basic_triangle
void vertex_particles(
    uint vertexId : SV_VertexID,
    out float4 position : SV_Position,
    out float3 color : COLOR
)
{
    // 创建一个简单的3x3网格
    float x = (float(vertexId % 3) - 1.0) * 0.5;
    float y = (float(vertexId / 3) - 1.0) * 0.5;
    
    position = float4(x, y, 0, 1);
    color = float3(1, 1, 1);
}

void pixel_particles(
    in float4 position : SV_Position,
    in float3 color : COLOR,
    out float4 outColor : SV_Target0
)
{
    outColor = float4(color, 1);
}