#include <donut/app/ApplicationBase.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/app/DeviceManager.h>
#include <donut/core/log.h>
#include <donut/core/vfs/VFS.h>
#include <nvrhi/utils.h>
#include <memory>
#include <chrono>
#include <filesystem>

using namespace donut;

static const char* g_WindowTitle = "Donut Example: Basic Compute Particles";

constexpr uint32_t c_MaxParticles = 1024;

struct Particle
{
    float position[3];
    float velocity[3];
    float life;
    uint32_t active;
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
    std::shared_ptr<engine::ShaderFactory> m_ShaderFactory;
    
    nvrhi::ComputePipelineHandle m_ComputePipeline;
    nvrhi::BufferHandle m_ParticleBuffer;
    nvrhi::BufferHandle m_ConstantBuffer;
    nvrhi::BindingLayoutHandle m_BindingLayout;
    nvrhi::BindingSetHandle m_BindingSet;
    
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
        CreatePipeline();
        
        m_PreviousTime = std::chrono::high_resolution_clock::now();
        
        return true;
    }
    
    void CreateResources()
    {
        nvrhi::BufferDesc particleBufferDesc;
        particleBufferDesc.byteSize = sizeof(Particle) * c_MaxParticles;
        particleBufferDesc.structStride = sizeof(Particle);
        particleBufferDesc.canHaveUAVs = true;
        particleBufferDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
        particleBufferDesc.keepInitialState = true;
        particleBufferDesc.debugName = "ParticleBuffer";
        m_ParticleBuffer = GetDevice()->createBuffer(particleBufferDesc);
        
        nvrhi::BufferDesc constantBufferDesc;
        constantBufferDesc.byteSize = sizeof(ComputeConstants);
        constantBufferDesc.isConstantBuffer = true;
        constantBufferDesc.isVolatile = true;
        constantBufferDesc.maxVersions = 16;
        constantBufferDesc.debugName = "ComputeConstants";
        m_ConstantBuffer = GetDevice()->createBuffer(constantBufferDesc);
        
        std::vector<Particle> initialParticles(c_MaxParticles);
        for (auto& particle : initialParticles)
        {
            particle.active = 0;
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
    
    void CreatePipeline()
    {
        auto computeShader = m_ShaderFactory->CreateShader("particles.hlsl", "compute_particles", nullptr, nvrhi::ShaderType::Compute);
        
        nvrhi::BindingLayoutDesc bindingLayoutDesc;
        bindingLayoutDesc.visibility = nvrhi::ShaderType::Compute;
        bindingLayoutDesc.bindings = {
            nvrhi::BindingLayoutItem::VolatileConstantBuffer(0),
            nvrhi::BindingLayoutItem::StructuredBuffer_UAV(0)
        };
        m_BindingLayout = GetDevice()->createBindingLayout(bindingLayoutDesc);
        
        nvrhi::ComputePipelineDesc pipelineDesc;
        pipelineDesc.CS = computeShader;
        pipelineDesc.bindingLayouts = { m_BindingLayout };
        m_ComputePipeline = GetDevice()->createComputePipeline(pipelineDesc);
        
        nvrhi::BindingSetDesc bindingSetDesc;
        bindingSetDesc.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(0, m_ConstantBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_UAV(0, m_ParticleBuffer)
        };
        m_BindingSet = GetDevice()->createBindingSet(bindingSetDesc, m_BindingLayout);
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
        
        nvrhi::ComputeState computeState;
        computeState.pipeline = m_ComputePipeline;
        computeState.bindings = { m_BindingSet };
        commandList->setComputeState(computeState);
        
        uint32_t dispatchX = (c_MaxParticles + 63) / 64;
        commandList->dispatch(dispatchX, 1, 1);
        
        commandList->clearState();
        
        nvrhi::utils::ClearColorAttachment(commandList, framebuffer, 0, nvrhi::Color(0.1f, 0.1f, 0.2f, 1.0f));
        
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