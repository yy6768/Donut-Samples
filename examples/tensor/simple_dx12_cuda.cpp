#include <donut/app/ApplicationBase.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/app/DeviceManager.h>
#include <donut/core/log.h>
#include <donut/core/vfs/VFS.h>
#include <donut/core/math/math.h>
#include <nvrhi/utils.h>
#include <memory>
#include <vector>
#include <chrono>
#include <filesystem>

// CUDA includes
#include <cuda_runtime.h>
#include <vector_types.h>
#include <aclapi.h>

// Windows security attributes for shared handles
class WindowsSecurityAttributes
{
protected:
    SECURITY_ATTRIBUTES  m_winSecurityAttributes;
    PSECURITY_DESCRIPTOR m_winPSecurityDescriptor;

public:
    WindowsSecurityAttributes();
    ~WindowsSecurityAttributes();
    SECURITY_ATTRIBUTES *operator&();
};

// WindowsSecurityAttributes implementation
WindowsSecurityAttributes::WindowsSecurityAttributes()
{
    m_winPSecurityDescriptor = (PSECURITY_DESCRIPTOR)calloc(1, SECURITY_DESCRIPTOR_MIN_LENGTH + 2 * sizeof(void **));
    assert(m_winPSecurityDescriptor != (PSECURITY_DESCRIPTOR)NULL);

    PSID *ppSID = (PSID *)((PBYTE)m_winPSecurityDescriptor + SECURITY_DESCRIPTOR_MIN_LENGTH);
    PACL *ppACL = (PACL *)((PBYTE)ppSID + sizeof(PSID *));

    InitializeSecurityDescriptor(m_winPSecurityDescriptor, SECURITY_DESCRIPTOR_REVISION);

    SID_IDENTIFIER_AUTHORITY sidIdentifierAuthority = SECURITY_WORLD_SID_AUTHORITY;
    AllocateAndInitializeSid(&sidIdentifierAuthority, 1, SECURITY_WORLD_RID, 0, 0, 0, 0, 0, 0, 0, ppSID);

    EXPLICIT_ACCESS explicitAccess;
    ZeroMemory(&explicitAccess, sizeof(EXPLICIT_ACCESS));
    explicitAccess.grfAccessPermissions = STANDARD_RIGHTS_ALL | SPECIFIC_RIGHTS_ALL;
    explicitAccess.grfAccessMode        = SET_ACCESS;
    explicitAccess.grfInheritance       = INHERIT_ONLY;
    explicitAccess.Trustee.TrusteeForm  = TRUSTEE_IS_SID;
    explicitAccess.Trustee.TrusteeType  = TRUSTEE_IS_WELL_KNOWN_GROUP;
    explicitAccess.Trustee.ptstrName    = (LPTSTR)*ppSID;

    SetEntriesInAcl(1, &explicitAccess, NULL, ppACL);
    SetSecurityDescriptorDacl(m_winPSecurityDescriptor, TRUE, *ppACL, FALSE);

    m_winSecurityAttributes.nLength              = sizeof(m_winSecurityAttributes);
    m_winSecurityAttributes.lpSecurityDescriptor = m_winPSecurityDescriptor;
    m_winSecurityAttributes.bInheritHandle       = TRUE;
}

WindowsSecurityAttributes::~WindowsSecurityAttributes()
{
    PSID *ppSID = (PSID *)((PBYTE)m_winPSecurityDescriptor + SECURITY_DESCRIPTOR_MIN_LENGTH);
    PACL *ppACL = (PACL *)((PBYTE)ppSID + sizeof(PSID *));

    if (*ppSID) {
        FreeSid(*ppSID);
    }
    if (*ppACL) {
        LocalFree(*ppACL);
    }
    free(m_winPSecurityDescriptor);
}

SECURITY_ATTRIBUTES *WindowsSecurityAttributes::operator&() { return &m_winSecurityAttributes; }

using namespace donut;

static const char* g_WindowTitle = "Donut Example: Simple DX12 CUDA Interop";

// Forward declare CUDA kernel
extern "C" void launch_simple_kernel(float4* d_data, unsigned int width, unsigned int height, float time);

class SimpleD3D12CudaApp : public app::IRenderPass
{
private:
    std::shared_ptr<vfs::RootFileSystem> m_RootFS;
    std::shared_ptr<engine::ShaderFactory> m_ShaderFactory;
    
    // D3D12 resources
    nvrhi::TextureHandle m_SharedTexture;
    nvrhi::GraphicsPipelineHandle m_GraphicsPipeline;
    nvrhi::ShaderHandle m_VertexShader;
    nvrhi::ShaderHandle m_PixelShader;
    nvrhi::BindingLayoutHandle m_BindingLayout;
    nvrhi::BindingSetHandle m_BindingSet;
    
    // CUDA resources
    float4* m_CudaLinearMemory = nullptr;
    
    // Synchronization
    cudaStream_t m_CudaStream = nullptr;
    
    // Animation
    std::chrono::high_resolution_clock::time_point m_StartTime;
    
    const uint32_t TEXTURE_WIDTH = 512;
    const uint32_t TEXTURE_HEIGHT = 512;

public:
    using IRenderPass::IRenderPass;
    
    bool Init()
    {
        log::info("=== SimpleD3D12CudaApp::Init() started ===");
        
        std::filesystem::path appShaderPath = app::GetDirectoryWithExecutable() / "shaders/tensor" / app::GetShaderTypeName(GetDevice()->getGraphicsAPI());
        
        auto nativeFS = std::make_shared<vfs::NativeFileSystem>();
        m_ShaderFactory = std::make_shared<engine::ShaderFactory>(GetDevice(), nativeFS, appShaderPath);
        
        m_StartTime = std::chrono::high_resolution_clock::now();
        
        if (!InitializeCuda()) {
            log::error("CUDA initialization failed");
            return false;
        }
        if (!CreateD3D12Resources()) {
            log::error("D3D12 resources creation failed");
            return false;
        }
        if (!CreateCudaInterop()) {
            log::error("CUDA interop creation failed");
            return false;
        }
        if (!CreateRenderPipeline()) {
            log::error("Render pipeline creation failed");
            return false;
        }
        
        log::info("=== SimpleD3D12CudaApp::Init() completed successfully ===");
        return true;
    }
    
    ~SimpleD3D12CudaApp()
    {
        // Clean up CUDA resources
        if (m_CudaLinearMemory)
        {
            cudaFree(m_CudaLinearMemory);
        }
        
        if (m_CudaStream)
        {
            cudaStreamDestroy(m_CudaStream);
        }
    }
    
private:
    bool InitializeCuda()
    {
        // Initialize CUDA
        cudaError_t cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess)
        {
            log::error("CUDA initialization failed: %s", cudaGetErrorString(cudaStatus));
            return false;
        }
        
        // Create CUDA stream
        cudaStatus = cudaStreamCreate(&m_CudaStream);
        if (cudaStatus != cudaSuccess)
        {
            log::error("Failed to create CUDA stream: %s", cudaGetErrorString(cudaStatus));
            return false;
        }
        
        log::info("CUDA initialized successfully");
        return true;
    }
    
    bool CreateD3D12Resources()
    {
        // Create shared texture
        nvrhi::TextureDesc textureDesc;
        textureDesc.width = TEXTURE_WIDTH;
        textureDesc.height = TEXTURE_HEIGHT;
        textureDesc.format = nvrhi::Format::RGBA32_FLOAT;
        textureDesc.initialState = nvrhi::ResourceStates::ShaderResource;
        textureDesc.keepInitialState = true;
        textureDesc.isUAV = true;
        textureDesc.isRenderTarget = false;
        textureDesc.debugName = "SharedTexture";
        m_SharedTexture = GetDevice()->createTexture(textureDesc);
        
        log::info("Shared texture created: %p", m_SharedTexture.Get());
        return true;
    }
    
    bool CreateCudaInterop()
    {
        // Allocate CUDA memory for now, can extend to true interop later
        size_t memSize = TEXTURE_WIDTH * TEXTURE_HEIGHT * sizeof(float4);
        cudaError_t cudaStatus = cudaMalloc((void**)&m_CudaLinearMemory, memSize);
        if (cudaStatus != cudaSuccess)
        {
            log::error("Failed to allocate CUDA memory: %s", cudaGetErrorString(cudaStatus));
            return false;
        }
        
        log::info("CUDA memory allocated successfully");
        return true;
    }
    
    bool CreateRenderPipeline()
    {
        // Create shaders
        m_VertexShader = m_ShaderFactory->CreateShader("simple_interop.hlsl", "vs_main", nullptr, nvrhi::ShaderType::Vertex);
        m_PixelShader = m_ShaderFactory->CreateShader("simple_interop.hlsl", "ps_main", nullptr, nvrhi::ShaderType::Pixel);
        
        if (!m_VertexShader || !m_PixelShader)
        {
            log::error("Failed to create shaders: VS=%p, PS=%p", m_VertexShader.Get(), m_PixelShader.Get());
            return false;
        }
        
        log::info("Shaders created successfully: VS=%p, PS=%p", m_VertexShader.Get(), m_PixelShader.Get());
        
        // Create binding layout for texture
        nvrhi::BindingLayoutDesc bindingLayoutDesc;
        bindingLayoutDesc.visibility = nvrhi::ShaderType::Pixel;
        bindingLayoutDesc.bindings = {
            nvrhi::BindingLayoutItem::Texture_SRV(0)
        };
        m_BindingLayout = GetDevice()->createBindingLayout(bindingLayoutDesc);
        
        log::info("Binding layout created: %p", m_BindingLayout.Get());
        
        return true;
    }
    
    void CreateGraphicsPipeline(nvrhi::IFramebuffer* framebuffer)
    {
        if (m_GraphicsPipeline) {
            log::info("Graphics pipeline already exists: %p", m_GraphicsPipeline.Get());
            return;
        }
        
        log::info("Creating graphics pipeline...");
            
        // Simplified version: no input layout needed, shader uses SV_VertexID
        nvrhi::GraphicsPipelineDesc psoDesc;
        psoDesc.VS = m_VertexShader;
        psoDesc.PS = m_PixelShader;
        psoDesc.primType = nvrhi::PrimitiveType::TriangleList;
        psoDesc.renderState.depthStencilState.depthTestEnable = false;
        psoDesc.bindingLayouts = { m_BindingLayout };
        
        m_GraphicsPipeline = GetDevice()->createGraphicsPipeline(psoDesc, framebuffer);
        log::info("Graphics pipeline created: %p", m_GraphicsPipeline.Get());
        
        // Create binding set  
        nvrhi::BindingSetDesc bindingSetDesc;
        bindingSetDesc.bindings = {
            nvrhi::BindingSetItem::Texture_SRV(0, m_SharedTexture)
        };
        m_BindingSet = GetDevice()->createBindingSet(bindingSetDesc, m_BindingLayout);
        log::info("Binding set created: %p", m_BindingSet.Get());
    }
    
public:
    void Animate(float fElapsedTimeSeconds) override
    {
        GetDeviceManager()->SetInformativeWindowTitle(g_WindowTitle);
    }
    
    void BackBufferResizing() override {}
    
    void Render(nvrhi::IFramebuffer* framebuffer) override
    {
        log::info("=== Render method called ===");
        log::info("Framebuffer provided: %p", framebuffer);
        
        CreateGraphicsPipeline(framebuffer);
        
        // Calculate animation time
        auto currentTime = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float>(currentTime - m_StartTime).count();
        
        log::info("Animation time: %.3f seconds", time);
        
        // Run CUDA processing
        ProcessWithCuda(time);
        
        // Render the result
        RenderResult(framebuffer);
        
        log::info("=== Render method completed ===");
    }
    
private:
    void ProcessWithCuda(float time)
    {
        // Launch CUDA kernel to generate procedural content
        launch_simple_kernel(m_CudaLinearMemory, TEXTURE_WIDTH, TEXTURE_HEIGHT, time);
        
        // Synchronize CUDA
        cudaStreamSynchronize(m_CudaStream);
        
        // Copy CUDA result to D3D12 texture (simplified approach)
        // CUDA generates float4 data, so we need to match that
        std::vector<float4> hostData(TEXTURE_WIDTH * TEXTURE_HEIGHT);
        cudaMemcpy(hostData.data(), m_CudaLinearMemory, 
                  TEXTURE_WIDTH * TEXTURE_HEIGHT * sizeof(float4), 
                  cudaMemcpyDeviceToHost);
        
        // Debug: Check if we have valid data (check center pixel instead of first)
        uint32_t center_idx = (TEXTURE_HEIGHT/2) * TEXTURE_WIDTH + (TEXTURE_WIDTH/2);
        log::info("First pixel: R=%.2f, G=%.2f, B=%.2f, A=%.2f", 
                  hostData[0].x, hostData[0].y, hostData[0].z, hostData[0].w);
        log::info("Center pixel: R=%.2f, G=%.2f, B=%.2f, A=%.2f", 
                  hostData[center_idx].x, hostData[center_idx].y, hostData[center_idx].z, hostData[center_idx].w);
        
        // Upload to D3D texture using command list
        auto commandList = GetDevice()->createCommandList();
        commandList->open();
        commandList->writeTexture(m_SharedTexture, 0, 0, hostData.data(), 
                                TEXTURE_WIDTH * sizeof(float4), TEXTURE_HEIGHT);
        commandList->close();
        GetDevice()->executeCommandList(commandList);
    }
    
    void RenderResult(nvrhi::IFramebuffer* framebuffer)
    {
        log::info("=== RenderResult called ===");
        log::info("Framebuffer: %p", framebuffer);
        
        auto commandList = GetDevice()->createCommandList();
        commandList->open();
        
        // Clear framebuffer to black to see if draw call works
        nvrhi::utils::ClearColorAttachment(commandList, framebuffer, 0, nvrhi::Color(0.0f, 0.0f, 0.0f, 1.0f));
        log::info("Framebuffer cleared to black");
        
        // Set graphics state (no vertex/index buffers needed)
        nvrhi::GraphicsState graphicsState;
        graphicsState.pipeline = m_GraphicsPipeline;
        graphicsState.framebuffer = framebuffer;
        graphicsState.bindings = { m_BindingSet };
        graphicsState.viewport.addViewportAndScissorRect(framebuffer->getFramebufferInfo().getViewport());
        
        commandList->setGraphicsState(graphicsState);
        log::info("Graphics state configured and set: pipeline=%p, bindings=%p", 
                 m_GraphicsPipeline.Get(), m_BindingSet.Get());
        
        // Add some checks to ensure state is correct
        if (!m_GraphicsPipeline) {
            log::error("Graphics pipeline is null!");
            return;
        }
        if (!m_BindingSet) {
            log::error("Binding set is null!");
            return;
        }
        
        log::info("About to create draw arguments");
        nvrhi::DrawArguments drawArgs;
        drawArgs.vertexCount = 3;
        drawArgs.instanceCount = 1;
        
        log::info("About to draw %d vertices", drawArgs.vertexCount);
        
        try {
            commandList->draw(drawArgs);
            log::info("Draw command issued successfully");
        } catch (...) {
            log::error("Exception occurred during draw call");
        }
        
        log::info("About to close command list");
        commandList->close();
        log::info("Command list closed, about to execute");
        GetDevice()->executeCommandList(commandList);
        
        log::info("=== RenderResult completed ===");
    }
};

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
    log::info("=== Application starting ===");
    
    nvrhi::GraphicsAPI api = app::GetGraphicsAPIFromCommandLine(__argc, __argv);
    
    // Only support D3D12 for this example
    if (api != nvrhi::GraphicsAPI::D3D12)
    {
        log::fatal("This example requires D3D12");
        return 1;
    }
    
    log::info("Using D3D12 graphics API");
    
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
    
    log::info("Creating window and device with resolution %dx%d", deviceParams.backBufferWidth, deviceParams.backBufferHeight);
    
    if (!deviceManager->CreateWindowDeviceAndSwapChain(deviceParams, g_WindowTitle))
    {
        log::fatal("Cannot initialize a graphics device with the requested parameters");
        return 1;
    }
    
    log::info("Device and window created successfully");
    
    {
        SimpleD3D12CudaApp app(deviceManager);
        if (app.Init())
        {
            log::info("Adding render pass to device manager");
            deviceManager->AddRenderPassToBack(&app);
            log::info("Starting message loop");
            deviceManager->RunMessageLoop();
            log::info("Message loop ended, removing render pass");
            deviceManager->RemoveRenderPass(&app);
        }
        else
        {
            log::fatal("Application initialization failed");
        }
    }
    
    log::info("Shutting down device manager");
    deviceManager->Shutdown();
    delete deviceManager;
    
    log::info("=== Application ended ===");
    return 0;
}