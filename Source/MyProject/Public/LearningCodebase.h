// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "LearningCodebase.generated.h"

/**
 * 
 */
UCLASS()
class MYPROJECT_API ULearningCodebase : public UBlueprintFunctionLibrary
{
	GENERATED_BODY()

public:
    
    UFUNCTION(BlueprintPure, Category="Observations")
    static TArray<AActor*> GetKClosest(AActor* Center, const TArray<AActor*>& Actors, int32 K);
};
