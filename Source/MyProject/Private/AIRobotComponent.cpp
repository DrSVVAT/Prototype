#include "AIRobotComponent.h"
#include "Math/UnrealMathUtility.h"

UAIRobotComponent::UAIRobotComponent()
{
    PrimaryComponentTick.bCanEverTick = true;

    // ������������� ����� ���������� ����������
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

float UAIRobotComponent::FeedForward(int32 TargetResourceIndex)
{
    // ������� ���� -> ������� ����
    TArray<float> HiddenLayer;
    HiddenLayer.SetNum(HiddenSize);

    for (int32 i = 0; i < HiddenSize; ++i)
    {
        // ������� ��������� �������� ����, ��������� ������ ������� ������
        float Activation = HiddenBiases[i];
        Activation += WeightsInputHidden[TargetResourceIndex * HiddenSize + i];
        HiddenLayer[i] = Sigmoid(Activation);
    }

    // ������� ���� -> �������� ����
    float Output = OutputBias;
    for (int32 i = 0; i < HiddenSize; ++i)
    {
        Output += WeightsHiddenOutput[i] * HiddenLayer[i];
    }

    return Sigmoid(Output);
}

float UAIRobotComponent::Predict(int32 TargetResourceIndex)
{
    return FeedForward(TargetResourceIndex);
}

void UAIRobotComponent::TrainNetwork(const TArray<FResourceDataStruct>& TrainingData)
{
    float LearningRate = 0.1f;

    // ������������ ������ ����� ������ ��� ��������
    for (const FResourceDataStruct& Data : TrainingData)
    {
        // ������ ������ ����� ��������� ����
        float PredictedOutput = FeedForward(Data.TargetResource);

        // ����������� ���������������� ������ (������������, ��� �������� = 10)
        float NormalizedRating = Data.UserRating / 10.0f;

        // ���������� ������ � ������ ���������������� ������
        float Error = NormalizedRating - PredictedOutput;
        float DeltaOutput = Error * PredictedOutput * (1 - PredictedOutput);

        // ���������� ����� ��������� ����
        TArray<float> HiddenLayer;
        HiddenLayer.SetNum(HiddenSize);

        // ���������� ��������� �������� ����
        for (int32 i = 0; i < HiddenSize; ++i)
        {
            float Activation = HiddenBiases[i];

            // ���� ������� ���� ��������� ��������
            for (int32 j = 0; j < Data.CollectedResources.Num(); ++j)
            {
                // �������� �� ����� ������� �� ������� �������
                if (j * HiddenSize + i < WeightsInputHidden.Num())
                {
                    Activation += WeightsInputHidden[j * HiddenSize + i] * Data.CollectedResources[j];
                }
            }

            // ���� �������� �������
            if (Data.CollectedResources.Num() * HiddenSize + i < WeightsInputHidden.Num())
            {
                Activation += WeightsInputHidden[Data.CollectedResources.Num() * HiddenSize + i] * Data.TargetResource;
            }

            // ��������� ������� ���������������� ������ �� ���������
            Activation += NormalizedRating;

            // ������� ���� -> ���������� ��������
            HiddenLayer[i] = Sigmoid(Activation);
            WeightsHiddenOutput[i] += LearningRate * DeltaOutput * HiddenLayer[i];
        }

        OutputBias += LearningRate * DeltaOutput;

        // ���������� ����� �������� ����
        for (int32 i = 0; i < HiddenSize; ++i)
        {
            float DeltaHidden = HiddenLayer[i] * (1 - HiddenLayer[i]) * WeightsHiddenOutput[i] * DeltaOutput;

            // ��������� ���� ��� ������� ���������� �������
            for (int32 j = 0; j < Data.CollectedResources.Num(); ++j)
            {
                // �������� �� ����� ������� �� ������� �������
                if (j * HiddenSize + i < WeightsInputHidden.Num())
                {
                    WeightsInputHidden[j * HiddenSize + i] += LearningRate * DeltaHidden * Data.CollectedResources[j];
                }
            }

            // ��������� ��� ��� �������� �������
            if (Data.CollectedResources.Num() * HiddenSize + i < WeightsInputHidden.Num())
            {
                WeightsInputHidden[Data.CollectedResources.Num() * HiddenSize + i] += LearningRate * DeltaHidden * Data.TargetResource;
            }

            // ��������� ������� ���������������� ������ �� ���������� ���� �������� ����
            HiddenBiases[i] += LearningRate * DeltaHidden * NormalizedRating;
        }
    }
}
