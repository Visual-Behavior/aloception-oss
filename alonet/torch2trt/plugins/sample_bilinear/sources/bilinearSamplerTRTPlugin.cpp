#include "bilinearSamplerTRTPlugin.h"
#include "NvInfer.h"
#include "bilinearSamplerKernel.h"

#include <NvInferRuntimeCommon.h>
#include <vector>
#include <cassert>
#include <cstring>
#include <iostream>

using namespace std;

#define assertm(exp, msg) assert(((void)msg, exp))

using namespace nvinfer1;

namespace {
    static const char* BILINEAR_SAMPLER_PLUGIN_VERSION{"1"};
    static const char* BILINEAR_SAMPLER_PLUGIN_NAME{"BilinearSampler_TRT"};
}

// Static class fields initialization
PluginFieldCollection BilinearSamplerPluginCreator::mFC{};
std::vector<PluginField> BilinearSamplerPluginCreator::mPluginAttributes;

// statically registers the Plugin Creator to the Plugin Registry of TensorRT
REGISTER_TENSORRT_PLUGIN(BilinearSamplerPluginCreator);

// Helper function for serializing plugin
template<typename T>
void writeToBuffer(char*& buffer, const T& val)
{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

// Helper function for deserializing plugin
template<typename T>
T readFromBuffer(const char*& buffer)
{
    T val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
    return val;
}

BilinearSamplerPlugin::BilinearSamplerPlugin(const std::string name)
    : mLayerName(name)
{
}

BilinearSamplerPlugin::BilinearSamplerPlugin(const std::string name, const void* data, size_t length)
    : mLayerName(name)
{
    // Deserialize in the same order as serialization
    const char *d = static_cast<const char *>(data);
    const char *a = d;

    mH = readFromBuffer<int>(d);
    mW = readFromBuffer<int>(d);
    mD = readFromBuffer<int>(d);

    assert(d == (a + length));
}

const char* BilinearSamplerPlugin::getPluginType() const noexcept
{
    return BILINEAR_SAMPLER_PLUGIN_NAME;
}

const char* BilinearSamplerPlugin::getPluginVersion() const noexcept
{
    return BILINEAR_SAMPLER_PLUGIN_VERSION;
}

int BilinearSamplerPlugin::getNbOutputs() const noexcept
{
    return 1;
}

Dims BilinearSamplerPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept
{
    // 2 inputs: image (B, H, W, 1) and coords (B, D, D, 2)
    assert(nbInputDims == 2);
    // only 1 output, so output index must be 0
    assert(index == 0);
    assert(inputs[0].nbDims == inputs[1].nbDims);
    cout << "BilinearSamplerPlugin inputs[0].nbDims " << inputs[0].nbDims << endl;
    cout << "BilinearSamplerPlugin inputs[0].d[...] " << inputs[0].d[0] << " " << inputs[0].d[1] << " " << inputs[0].d[2] << endl;
    cout << "BilinearSamplerPlugin inputs[0].d[...] " << inputs[1].d[0] << " " << inputs[1].d[1] << " " << inputs[1].d[2] << endl;
    assert(inputs[0].nbDims == 3);
    assert(inputs[1].d[0] == inputs[1].d[1]);
    // Dims ret = inputs[1];
    // ret.d[3] = 1;
    return Dims3(inputs[1].d[0], inputs[1].d[1], 1);
}

int BilinearSamplerPlugin::initialize() noexcept
{
    return 0;
}

int BilinearSamplerPlugin::enqueue(int batchSize, const void* const* inputs, void* const* outputs,
    void* workspace, cudaStream_t stream) noexcept
{
    int status = -1;

    // Our plugin outputs only one tensor
    // void* output = outputs[0];

    // Launch CUDA kernel wrapper and save its return value
    // status = custom_bilinear_sampler(stream, mB, mH, mW, mD, inputs[0], inputs[1], output);
    status = custom_bilinear_sampler(stream, batchSize, mH, mW, mD, inputs[0], inputs[1], outputs[0]);
    assert(status == 0);
    return status;
}

size_t BilinearSamplerPlugin::getSerializationSize() const noexcept
{
    // int parameters: mH, mW, mD
    return 3 * sizeof(int32_t);
}

void BilinearSamplerPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char *>(buffer);
    const char *a = d;

    // writeToBuffer(d, mB);
    writeToBuffer(d, mH);
    writeToBuffer(d, mW);
    writeToBuffer(d, mD);

    assert(d == a + getSerializationSize());
}

void BilinearSamplerPlugin::terminate() noexcept {}

void BilinearSamplerPlugin::destroy() noexcept {
    // This gets called when the network containing plugin is destroyed
    delete this;
}

DataType BilinearSamplerPlugin::getOutputDataType(int32_t index, const DataType *inputTypes, int32_t nbInputs) const noexcept
{
    // only 1 output
    assert(index == 0);
    assert(inputTypes && nbInputs > 0);
    return inputTypes[0]; // return type of input tensor image
}

bool BilinearSamplerPlugin::isOutputBroadcastAcrossBatch(int32_t outputIndex,
    const bool* inputIsBroadcasted, int32_t nbInputs) const noexcept
{
    return false;
}

bool BilinearSamplerPlugin::canBroadcastInputAcrossBatch(int inputIndex) const noexcept
{
    return false;
}

void BilinearSamplerPlugin::configurePlugin(const PluginTensorDesc* in, int32_t nbInput,
    const PluginTensorDesc* out, int32_t nbOutput) noexcept
{
    assertm(nbInput == 2, "Must provide 2 inputs: image (B, H, W, 1) and coords (B, D, D, 2)");
    assertm(nbOutput == 1, "This layer has only one output.");
    Dims image_dims = in[0].dims;
    Dims coords_dims = in[1].dims;
    // cout for debug
    cout << image_dims.nbDims << "\t" << coords_dims.nbDims << endl;
    cout << image_dims.d[0] << endl;
    cout << coords_dims.d[0] << endl;
    assertm(image_dims.nbDims == 3, "image input must have rank 3 (H, W, 1)");
    assertm(coords_dims.nbDims == 3, "coords input must have rank (D, D, 2)");
    // assertm(image_dims.d[0] == coords_dims.d[0], "image and coords first dimesion must be equal.\n");
    assertm(coords_dims.d[0] == coords_dims.d[1], "coords of shape (batch, D, D, 2)\n");
    assertm(coords_dims.d[2] == 2, "coords of shape (B, D, D, 2)\n");
    // mB = image_dims.d[0];
    mH = image_dims.d[0];
    mW = image_dims.d[1];
    mD = coords_dims.d[0];
}

bool BilinearSamplerPlugin::supportsFormatCombination(int pos, const PluginTensorDesc* inOut,
    int nbInputs, int nbOutputs) const noexcept
{

    return (inOut[pos].format == TensorFormat::kLINEAR) && (inOut[pos].type == DataType::kFLOAT)
        && (inOut[pos].format == inOut[0].format);
}

IPluginV2Ext* BilinearSamplerPlugin::clone() const noexcept
{
    auto plugin = new BilinearSamplerPlugin(mLayerName);
    // plugin->mB = mB;
    plugin->mH = mH;
    plugin->mW = mW;
    plugin->mD = mD;
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

void BilinearSamplerPlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* BilinearSamplerPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

BilinearSamplerPluginCreator::BilinearSamplerPluginCreator()
{
    // Describe BilinearSamplerPlugin's required PluginField arguments

    // Fill PluginFieldCollection with PluginField arguments metadata
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* BilinearSamplerPluginCreator::getPluginName() const noexcept
{
    return BILINEAR_SAMPLER_PLUGIN_NAME;
}

const char* BilinearSamplerPluginCreator::getPluginVersion() const noexcept
{
    return BILINEAR_SAMPLER_PLUGIN_VERSION;
}

const PluginFieldCollection* BilinearSamplerPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* BilinearSamplerPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    // const PluginField* fields = fc->fields;

    // Parse fields from PluginFieldCollection
    return new BilinearSamplerPlugin(name);
}

IPluginV2* BilinearSamplerPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call BilinearSamplerPlugin::destroy()
    return new BilinearSamplerPlugin(name, serialData, serialLength);
}

void BilinearSamplerPluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* BilinearSamplerPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}
