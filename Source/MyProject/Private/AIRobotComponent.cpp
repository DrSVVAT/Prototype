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

float UAIRobotComponent::FeedForward(int32 TargetResourceIndex)
{
    // Входной слой -> скрытый слой
    TArray<float> HiddenLayer;
    HiddenLayer.SetNum(HiddenSize);

    for (int32 i = 0; i < HiddenSize; ++i)
    {
        // Считаем активацию скрытого слоя, используя только целевой ресурс
        float Activation = HiddenBiases[i];
        Activation += WeightsInputHidden[TargetResourceIndex * HiddenSize + i];
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

float UAIRobotComponent::Predict(int32 TargetResourceIndex)
{
    return FeedForward(TargetResourceIndex);
}

void UAIRobotComponent::TrainNetwork(const TArray<FResourceDataStruct>& TrainingData)
{
    float LearningRate = 0.1f;

    // Обрабатываем каждый набор данных для обучения
    for (const FResourceDataStruct& Data : TrainingData)
    {
        // Прямой проход через нейронную сеть
        float PredictedOutput = FeedForward(Data.TargetResource);

        // Нормализуем пользовательскую оценку (предполагаем, что максимум = 10)
        float NormalizedRating = Data.UserRating / 10.0f;

        // Вычисление ошибки с учётом пользовательской оценки
        float Error = NormalizedRating - PredictedOutput;
        float DeltaOutput = Error * PredictedOutput * (1 - PredictedOutput);

        // Обновление весов выходного слоя
        TArray<float> HiddenLayer;
        HiddenLayer.SetNum(HiddenSize);

        // Вычисление активации скрытого слоя
        for (int32 i = 0; i < HiddenSize; ++i)
        {
            float Activation = HiddenBiases[i];

            // Учёт влияния всех собранных ресурсов
            for (int32 j = 0; j < Data.CollectedResources.Num(); ++j)
            {
                // Проверка на выход индекса за пределы массива
                if (j * HiddenSize + i < WeightsInputHidden.Num())
                {
                    Activation += WeightsInputHidden[j * HiddenSize + i] * Data.CollectedResources[j];
                }
            }

            // Учёт целевого ресурса
            if (Data.CollectedResources.Num() * HiddenSize + i < WeightsInputHidden.Num())
            {
                Activation += WeightsInputHidden[Data.CollectedResources.Num() * HiddenSize + i] * Data.TargetResource;
            }

            // Добавляем влияние пользовательской оценки на активацию
            Activation += NormalizedRating;

            // Скрытый слой -> вычисление значения
            HiddenLayer[i] = Sigmoid(Activation);
            WeightsHiddenOutput[i] += LearningRate * DeltaOutput * HiddenLayer[i];
        }

        OutputBias += LearningRate * DeltaOutput;

        // Обновление весов скрытого слоя
        for (int32 i = 0; i < HiddenSize; ++i)
        {
            float DeltaHidden = HiddenLayer[i] * (1 - HiddenLayer[i]) * WeightsHiddenOutput[i] * DeltaOutput;

            // Обновляем веса для каждого собранного ресурса
            for (int32 j = 0; j < Data.CollectedResources.Num(); ++j)
            {
                // Проверка на выход индекса за пределы массива
                if (j * HiddenSize + i < WeightsInputHidden.Num())
                {
                    WeightsInputHidden[j * HiddenSize + i] += LearningRate * DeltaHidden * Data.CollectedResources[j];
                }
            }

            // Обновляем вес для целевого ресурса
            if (Data.CollectedResources.Num() * HiddenSize + i < WeightsInputHidden.Num())
            {
                WeightsInputHidden[Data.CollectedResources.Num() * HiddenSize + i] += LearningRate * DeltaHidden * Data.TargetResource;
            }

            // Добавляем влияние пользовательской оценки на обновление веса скрытого слоя
            HiddenBiases[i] += LearningRate * DeltaHidden * NormalizedRating;
        }
    }
}
