#include <donut/app/ApplicationBase.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/app/DeviceManager.h>
#include <donut/core/log.h>
#include <donut/core/vfs/VFS.h>
#include <donut/core/math/math.h>
#include <nvrhi/utils.h>
#include <memory>
#include <chrono>
#include <filesystem>

#include "particles_cb.h"

using namespace donut;
using namespace donut::math;

static const char* g_WindowTitle = "Donut Example: Basic Compute Particles";

constexpr uint32_t c_MaxParticles = 1024;

static float RandomFloat()
{
    return float(std::rand()) / RAND_MAX;
}

static float3 RandomFloat3()
{
    return float3(RandomFloat(), RandomFloat(), RandomFloat());
}

struct Particle
{
    float2 position;
    float2 velocity;
    float3 color;
};



struct ComputeConstants
{
    float deltaTime;
    uint32_t particleCount;
    float gravity;
    float lifeTime;
    float padding[4];
};

class BasicComputeApp : public app::IRenderPass
{
private:
	std::shared_ptr<vfs::RootFileSystem> m_RootFS;
    // Compute pipeline
    nvrhi::ShaderHandle m_ComputeShader;
    nvrhi::ComputePipelineHandle m_ComputePipeline;
    nvrhi::CommandListHandle m_CommandList;
    nvrhi::BindingLayoutHandle m_ComputeBindingLayout;
    nvrhi::BindingLayoutHandle m_GraphicsBindingLayout;
    nvrhi::BindingSetHandle m_ComputeBindingSet;
    nvrhi::BindingSetHandle m_GraphicsBindingSet;

    // CB
    nvrhi::BufferHandle m_ConstantBuffer;

    // Particle buffer
    std::vector<Particle> m_Particles;
    nvrhi::BufferHandle m_ParticleBuffer;

    std::shared_ptr<engine::ShaderFactory> m_ShaderFactory;

    nvrhi::ShaderHandle m_VertexShader;
    nvrhi::ShaderHandle m_PixelShader;
    
    nvrhi::GraphicsPipelineHandle m_GraphicsPipeline;
    
    
    std::chrono::high_resolution_clock::time_point m_PreviousTime;
    
    float m_Gravity = 9.8f;
    float m_LifeTime = 5.0f;

public:
    using IRenderPass::IRenderPass;
    bool Init()
    {
        std::filesystem::path appShaderPath = app::GetDirectoryWithExecutable() / "shaders/basic_compute" / app::GetShaderTypeName(GetDevice()->getGraphicsAPI());
        
        auto nativeFS = std::make_shared<vfs::NativeFileSystem>();
        m_ShaderFactory = std::make_shared<engine::ShaderFactory>(GetDevice(), nativeFS, appShaderPath);
        
        CreateResources();
        CreateComputePipeline();
        
        m_PreviousTime = std::chrono::high_resolution_clock::now();
        
        return true;
    }
    
    void CreateResources()
    {
        // Particle buffer - 需要同时支持UAV (compute)和SRV (graphics)
        nvrhi::BufferDesc particleBufferDesc;
        particleBufferDesc.byteSize = sizeof(Particle) * c_MaxParticles;
        particleBufferDesc.structStride = sizeof(Particle);
        particleBufferDesc.canHaveUAVs = true;
        particleBufferDesc.canHaveRawViews = true;
        particleBufferDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
        particleBufferDesc.keepInitialState = true;
        particleBufferDesc.debugName = "ParticleBuffer";
        m_ParticleBuffer = GetDevice()->createBuffer(particleBufferDesc);
        
        // Constant buffer
        nvrhi::BufferDesc constantBufferDesc;
        constantBufferDesc.byteSize = sizeof(ComputeConstants);
        constantBufferDesc.isConstantBuffer = true;
        constantBufferDesc.isVolatile = true;
        constantBufferDesc.maxVersions = 16;
        constantBufferDesc.debugName = "ComputeConstants";
        m_ConstantBuffer = GetDevice()->createBuffer(constantBufferDesc);

        // Shader handles
        m_VertexShader = m_ShaderFactory->CreateShader("particles.hlsl", "vertex_particles", nullptr, nvrhi::ShaderType::Vertex);
        m_PixelShader = m_ShaderFactory->CreateShader("particles.hlsl", "pixel_particles", nullptr, nvrhi::ShaderType::Pixel);
        m_ComputeShader = m_ShaderFactory->CreateShader("particles.hlsl", "compute_particles", nullptr, nvrhi::ShaderType::Compute);
        
        // init particles
        std::vector<Particle> initialParticles(c_MaxParticles);
        for (uint32_t i = 0; i < c_MaxParticles; ++i)
        {
            initialParticles[i].position = math::float2(RandomFloat() * 2.0f - 1.0f, RandomFloat() * 2.0f - 1.0f);
            initialParticles[i].velocity = math::float2((RandomFloat() - 0.5f) * 0.1f, (RandomFloat() - 0.5f) * 0.1f);
            initialParticles[i].color = math::float3(RandomFloat(), RandomFloat(), RandomFloat());
        }
        
        auto commandList = GetDevice()->createCommandList();
        commandList->open();
        commandList->beginTrackingBufferState(m_ParticleBuffer, nvrhi::ResourceStates::UnorderedAccess);
        commandList->setBufferState(m_ParticleBuffer, nvrhi::ResourceStates::CopyDest);
        commandList->writeBuffer(m_ParticleBuffer, initialParticles.data(), initialParticles.size() * sizeof(Particle));
        commandList->setBufferState(m_ParticleBuffer, nvrhi::ResourceStates::UnorderedAccess);
        commandList->close();
        GetDevice()->executeCommandList(commandList);
        GetDevice()->waitForIdle();
    }
    
    void CreateComputePipeline()
    {
        // Compute binding layout
        nvrhi::BindingLayoutDesc computeBindingLayoutDesc;
        computeBindingLayoutDesc.visibility = nvrhi::ShaderType::Compute;
        computeBindingLayoutDesc.bindings = {
            nvrhi::BindingLayoutItem::VolatileConstantBuffer(0),
            nvrhi::BindingLayoutItem::StructuredBuffer_UAV(0)
        };
        m_ComputeBindingLayout = GetDevice()->createBindingLayout(computeBindingLayoutDesc);
        
        nvrhi::ComputePipelineDesc pipelineDesc;
        pipelineDesc.CS = m_ComputeShader;
        pipelineDesc.bindingLayouts = { m_ComputeBindingLayout };
        m_ComputePipeline = GetDevice()->createComputePipeline(pipelineDesc);
        
        nvrhi::BindingSetDesc computeBindingSetDesc;
        computeBindingSetDesc.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(0, m_ConstantBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_UAV(0, m_ParticleBuffer)
        };
        m_ComputeBindingSet = GetDevice()->createBindingSet(computeBindingSetDesc, m_ComputeBindingLayout);
    }

    void CreateGraphicsPipeline(nvrhi::IFramebuffer* framebuffer) 
    {
        if (m_GraphicsPipeline)
            return;
            
        // 检查shader是否加载成功
        if (!m_VertexShader || !m_PixelShader)
        {
            log::error("Failed to load vertex or pixel shader");
            return;
        }
            
        // Graphics binding layout - 需要constant buffer和SRV
        nvrhi::BindingLayoutDesc graphicsBindingLayoutDesc;
        graphicsBindingLayoutDesc.visibility = nvrhi::ShaderType::Vertex | nvrhi::ShaderType::Pixel;
        graphicsBindingLayoutDesc.bindings = {
            nvrhi::BindingLayoutItem::VolatileConstantBuffer(0),  // b0
            nvrhi::BindingLayoutItem::StructuredBuffer_SRV(0)    // t0
        };
        m_GraphicsBindingLayout = GetDevice()->createBindingLayout(graphicsBindingLayoutDesc);
        
        // Graphics binding set
        nvrhi::BindingSetDesc graphicsBindingSetDesc;
        graphicsBindingSetDesc.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(0, m_ConstantBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(0, m_ParticleBuffer)
        };
        m_GraphicsBindingSet = GetDevice()->createBindingSet(graphicsBindingSetDesc, m_GraphicsBindingLayout);
        
        // Graphics pipeline - 使用SV_VertexID，不需要input layout
        nvrhi::GraphicsPipelineDesc psoDesc;
        psoDesc.VS = m_VertexShader;
        psoDesc.PS = m_PixelShader;
        psoDesc.inputLayout = nullptr;  // 不使用input layout
        psoDesc.primType = nvrhi::PrimitiveType::PointList;
        psoDesc.renderState.depthStencilState.depthTestEnable = false;
        psoDesc.renderState.blendState.targets[0].blendEnable = true;
        psoDesc.renderState.blendState.targets[0].srcBlend = nvrhi::BlendFactor::SrcAlpha;
        psoDesc.renderState.blendState.targets[0].destBlend = nvrhi::BlendFactor::InvSrcAlpha;
        psoDesc.renderState.blendState.targets[0].blendOp = nvrhi::BlendOp::Add;
        psoDesc.bindingLayouts = { m_GraphicsBindingLayout };

        m_GraphicsPipeline = GetDevice()->createGraphicsPipeline(psoDesc, framebuffer);
        
        if (!m_GraphicsPipeline)
        {
            log::error("Failed to create graphics pipeline");
        }
    }
    void Animate(float fElapsedTimeSeconds) override
    {
        GetDeviceManager()->SetInformativeWindowTitle(g_WindowTitle);
    }
    
    void BackBufferResizing() override
    {
    }
    
    void Render(nvrhi::IFramebuffer* framebuffer) override
    {
        if (!m_GraphicsPipeline) {
            CreateGraphicsPipeline(framebuffer);
        }
        auto currentTime = std::chrono::high_resolution_clock::now();
        float deltaTime = std::chrono::duration<float>(currentTime - m_PreviousTime).count();
        m_PreviousTime = currentTime;
        
        ComputeConstants constants = {};
        constants.deltaTime = deltaTime;
        constants.particleCount = c_MaxParticles;
        constants.gravity = m_Gravity;
        constants.lifeTime = m_LifeTime;
        
        auto commandList = GetDevice()->createCommandList();
        commandList->open();
        
        commandList->writeBuffer(m_ConstantBuffer, &constants, sizeof(constants));
        
        // Compute pass
        nvrhi::ComputeState computeState;
        computeState.pipeline = m_ComputePipeline;
        computeState.bindings = { m_ComputeBindingSet };
        commandList->setComputeState(computeState);
        
        uint32_t dispatchX = (c_MaxParticles + 63) / 64;
        commandList->dispatch(dispatchX, 1, 1);
        
        // 同步：compute -> graphics
        commandList->setBufferState(m_ParticleBuffer, nvrhi::ResourceStates::ShaderResource);
        
        // Clear framebuffer
        nvrhi::utils::ClearColorAttachment(commandList, framebuffer, 0, nvrhi::Color(0.1f, 0.1f, 0.2f, 1.0f));
        
        // Graphics pass
        nvrhi::GraphicsState graphicsState;
        graphicsState.pipeline = m_GraphicsPipeline;
        graphicsState.framebuffer = framebuffer;
        graphicsState.bindings = { m_GraphicsBindingSet };
        graphicsState.viewport.addViewportAndScissorRect(framebuffer->getFramebufferInfo().getViewport());
        // 不设置vertex buffers，使用SV_VertexID
        
        commandList->setGraphicsState(graphicsState);
        
        nvrhi::DrawArguments args;
        args.vertexCount = c_MaxParticles;
        commandList->draw(args);

        // 恢复状态用于下一帧compute
        commandList->setBufferState(m_ParticleBuffer, nvrhi::ResourceStates::UnorderedAccess);

        commandList->close();
        GetDevice()->executeCommandList(commandList);
    }
};

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
    nvrhi::GraphicsAPI api = app::GetGraphicsAPIFromCommandLine(__argc, __argv);
    app::DeviceManager* deviceManager = app::DeviceManager::Create(api);
    
    app::DeviceCreationParameters deviceParams;
    deviceParams.backBufferWidth = 1280;
    deviceParams.backBufferHeight = 720;
    deviceParams.swapChainBufferCount = 3;
    deviceParams.swapChainFormat = nvrhi::Format::SRGBA8_UNORM;
    deviceParams.refreshRate = 0;
    deviceParams.vsyncEnabled = true;
    deviceParams.enableDebugRuntime = false;
    deviceParams.enableNvrhiValidationLayer = false;
    
    if (!deviceManager->CreateWindowDeviceAndSwapChain(deviceParams, g_WindowTitle))
    {
        log::fatal("Cannot initialize a graphics device with the requested parameters");
        return 1;
    }
    
    {
        BasicComputeApp app(deviceManager);
        if (app.Init())
        {
            deviceManager->AddRenderPassToBack(&app);
            deviceManager->RunMessageLoop();
            deviceManager->RemoveRenderPass(&app);
        }
    }
    
    deviceManager->Shutdown();
    delete deviceManager;
    
    return 0;
}