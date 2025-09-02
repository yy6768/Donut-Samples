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
    float3 position;
    float3 velocity;
    float life;
    uint active;
};

RWStructuredBuffer<Particle> g_ParticleBuffer : register(u0);

[numthreads(64, 1, 1)]
void compute_particles(uint3 threadId : SV_DispatchThreadID)
{
    uint index = threadId.x;
    if (index >= g_ParticleCount)
        return;
    
    Particle particle = g_ParticleBuffer[index];
    
    if (particle.active == 0)
    {
        // Respawn particle
        particle.position = float3(0, 5, 0);
        particle.velocity = float3(
            (frac(sin(dot(float2(index, index * 1.3), float2(12.9898, 78.233))) * 43758.5453) - 0.5) * 2.0,
            abs(frac(sin(dot(float2(index * 2.1, index), float2(12.9898, 78.233))) * 43758.5453)) * 3.0 + 1.0,
            (frac(sin(dot(float2(index * 0.7, index * 1.7), float2(12.9898, 78.233))) * 43758.5453) - 0.5) * 2.0
        );
        particle.life = g_LifeTime;
        particle.active = 1;
    }
    else
    {
        // Update particle
        particle.velocity.y -= g_Gravity * g_DeltaTime;
        particle.position += particle.velocity * g_DeltaTime;
        particle.life -= g_DeltaTime;
        
        if (particle.life <= 0.0 || particle.position.y < -10.0)
        {
            particle.active = 0;
        }
    }
    
    g_ParticleBuffer[index] = particle;
}