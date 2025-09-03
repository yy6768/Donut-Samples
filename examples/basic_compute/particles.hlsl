cbuffer Constants : register(b0)
{
    float g_DeltaTime;
    uint g_ParticleCount;
    float g_Gravity;
    float g_LifeTime;
    float4 g_Padding;
};

struct Particle
{
    float2 position;
    float2 velocity;
    float3 color;
};

RWStructuredBuffer<Particle> g_ParticleBuffer : register(u0);
StructuredBuffer<Particle> g_ParticleBufferSRV : register(t0);

struct VSOutput
{
    float4 position : SV_Position;
    float3 color    : COLOR;
};

// 顶点着色器 - 简化版本，直接从particle buffer读取
VSOutput vertex_particles(uint vertexId : SV_VertexID)
{
    VSOutput output;
    
    if (vertexId >= g_ParticleCount)
    {
        output.position = float4(0, 0, 0, 0);
        output.color = float3(0, 0, 0);
        return output;
    }
    
    Particle particle = g_ParticleBufferSRV[vertexId];
    
    output.position = float4(particle.position, 0.0, 1.0);
    output.color = particle.color;
    
    return output;
}

float4 pixel_particles(VSOutput input) : SV_Target
{
    return float4(input.color, 1.0);
}

[numthreads(64, 1, 1)]
void compute_particles(uint3 threadId : SV_DispatchThreadID)
{
    uint index = threadId.x;
    if (index >= g_ParticleCount)
        return;
    
    Particle particle = g_ParticleBuffer[index];
    
    // Apply gravity
    particle.velocity.y -= g_Gravity * g_DeltaTime * 0.01; // Scale down for 2D
    
    // Update position
    particle.position += particle.velocity * g_DeltaTime;
    
    // Bounce off edges
    if (particle.position.x < -1.0 || particle.position.x > 1.0)
    {
        particle.velocity.x *= -0.8;
        particle.position.x = clamp(particle.position.x, -1.0, 1.0);
    }
    if (particle.position.y < -1.0)
    {
        particle.velocity.y *= -0.8;
        particle.position.y = -1.0;
    }
    if (particle.position.y > 1.0)
    {
        particle.velocity.y *= -0.8;
        particle.position.y = 1.0;
    }
    
    // Add some color animation
    float time = g_DeltaTime * index * 0.01;
    particle.color.r = 0.5 + 0.5 * sin(time);
    particle.color.g = 0.5 + 0.5 * cos(time * 1.3);
    particle.color.b = 0.5 + 0.5 * sin(time * 0.7);
    
    g_ParticleBuffer[index] = particle;
}