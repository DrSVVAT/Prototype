#include "AIRobotComponent.h"
#include "Math/UnrealMathUtility.h"

UAIRobotComponent::UAIRobotComponent()
{
    PrimaryComponentTick.bCanEverTick = true;

    // Инициализация весов случайными значениями
    WeightsInputHidden.SetNum(InputSize * HiddenSize);
    WeightsHiddenOutput.SetNum(HiddenSize);
    HiddenBiases.SetNum(HiddenSize);

    for (float& Weight : WeightsInputHidden)
    {
        Weight = FMath::FRandRange(-1.0f, 1.0f);
    }

    for (float& Weight : WeightsHiddenOutput)
    {
        Weight = FMath::FRandRange(-1.0f, 1.0f);
    }

    for (float& Bias : HiddenBiases)
    {
        Bias = FMath::FRandRange(-1.0f, 1.0f);
    }

    OutputBias = FMath::FRandRange(-1.0f, 1.0f);
}

void UAIRobotComponent::BeginPlay()
{
    Super::BeginPlay();
}

float UAIRobotComponent::Sigmoid(float Value)
{
    return 1.0f / (1.0f + FMath::Exp(-Value));
}

float UAIRobotComponent::FeedForward(int32 ResourceIndex)
{
    // Входной слой -> скрытый слой
    TArray<float> HiddenLayer;
    HiddenLayer.SetNum(HiddenSize);

    for (int32 i = 0; i < HiddenSize; ++i)
    {
        float Activation = HiddenBiases[i];
        Activation += WeightsInputHidden[0 * HiddenSize + i] * ResourceIndex;

        HiddenLayer[i] = Sigmoid(Activation);
    }

    // Скрытый слой -> выходной слой
    float Output = OutputBias;
    for (int32 i = 0; i < HiddenSize; ++i)
    {
        Output += WeightsHiddenOutput[i] * HiddenLayer[i];
    }

    return Sigmoid(Output);
}

float UAIRobotComponent::Predict(int32 ResourceIndex)
{
    return FeedForward(ResourceIndex);
}

void UAIRobotComponent::TrainNetwork(const TArray<FResourceDataStruct>& TrainingData)
{
    float LearningRate = 0.1f;

    for (const FResourceDataStruct& Data : TrainingData)
    {
        // Прямой проход
        float PredictedOutput = FeedForward(Data.TargetResource);

        // Вычисление ошибки
        float Error = Data.UserRating - PredictedOutput;
        float DeltaOutput = Error * PredictedOutput * (1 - PredictedOutput);

        // Обновление весов выходного слоя
        TArray<float> HiddenLayer;
        HiddenLayer.SetNum(HiddenSize);

        for (int32 i = 0; i < HiddenSize; ++i)
        {
            float Activation = HiddenBiases[i];
            Activation += WeightsInputHidden[0 * HiddenSize + i] * Data.TargetResource;

            HiddenLayer[i] = Sigmoid(Activation);
            WeightsHiddenOutput[i] += LearningRate * DeltaOutput * HiddenLayer[i];
        }
        OutputBias += LearningRate * DeltaOutput;

        // Обновление весов скрытого слоя
        for (int32 i = 0; i < HiddenSize; ++i)
        {
            float DeltaHidden = HiddenLayer[i] * (1 - HiddenLayer[i]) * WeightsHiddenOutput[i] * DeltaOutput;
            WeightsInputHidden[0 * HiddenSize + i] += LearningRate * DeltaHidden * Data.TargetResource;
            HiddenBiases[i] += LearningRate * DeltaHidden;
        }
    }
}
