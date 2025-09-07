// 简化版本：不使用顶点缓冲区，在着色器中生成全屏四边形
static const float2 g_positions[] = {
    float2(-1.0, -1.0),  // 左下
    float2( 3.0, -1.0),  // 右下（超出屏幕以覆盖整个屏幕）
    float2(-1.0,  3.0)   // 左上（超出屏幕以覆盖整个屏幕）
};

static const float2 g_texcoords[] = {
    float2(0.0, 1.0),  // 左下
    float2(2.0, 1.0),  // 右下
    float2(0.0, -1.0)  // 左上
};

struct VS_OUTPUT
{
    float4 position : SV_Position;
    float2 texcoord : TEXCOORD0;
};

Texture2D<float4> g_texture : register(t0);

VS_OUTPUT vs_main(uint vertexId : SV_VertexID)
{
    VS_OUTPUT output;
    output.position = float4(g_positions[vertexId], 0.0f, 1.0f);
    output.texcoord = g_texcoords[vertexId];
    return output;
}

float4 ps_main(VS_OUTPUT input) : SV_Target
{
    // 强制返回明亮的红色来测试像素着色器是否工作
    return float4(1.0, 0.0, 0.0, 1.0); // 亮红色
}
