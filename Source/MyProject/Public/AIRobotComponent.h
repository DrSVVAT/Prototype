// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "AIRobotComponent.generated.h"

// Структура данных для обучения
USTRUCT(BlueprintType)
struct FResourceDataStruct
{
	GENERATED_BODY()

	UPROPERTY(BlueprintReadWrite)
	float TargetResource;

	UPROPERTY(BlueprintReadWrite)
	TArray<float> CollectedResources;

	UPROPERTY(BlueprintReadWrite)
	float UserRating;
};


UCLASS( ClassGroup=(Custom), meta=(BlueprintSpawnableComponent) )
class MYPROJECT_API UAIRobotComponent : public UActorComponent
{
	GENERATED_BODY()

public:	
	// Sets default values for this component's properties
	UAIRobotComponent();

protected:
	// Called when the game starts
	virtual void BeginPlay() override;

public:	
	UFUNCTION(BlueprintCallable, Category = "AI Robot")
	float Predict(int32 ResourceIndex);

	UFUNCTION(BlueprintCallable, Category = "AI Robot")
	void TrainNetwork(const TArray<FResourceDataStruct>& TrainingData);

private:
	const int32 InputSize = 3;
	const int32 HiddenSize = 5;
	const int32 OutputSize = 1;

	TArray<float> WeightsInputHidden;
	TArray<float> WeightsHiddenOutput;
	TArray<float> HiddenBiases;
	float OutputBias;

	float Sigmoid(float Value);
	float FeedForward(int32 ResourceIndex);

};
