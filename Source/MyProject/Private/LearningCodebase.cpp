// Fill out your copyright notice in the Description page of Project Settings.


#include "LearningCodebase.h"
#include "GameFramework/Actor.h"
#include "Kismet/KismetMathLibrary.h"

TArray<AActor*> ULearningCodebase::GetKClosest(AActor* Center, const TArray<AActor*>& Actors, int32 K)
{
    if (K < 0) {
        K = 0;
    }
    
    TArray<AActor*> Result = Actors;
    Result.RemoveAll([](AActor* Actor) { return Actor == nullptr; });
    
    if (Center) {
        Result.Sort([&](const AActor& A, const AActor& B)
                    {
            return FVector::DistSquared(Center->GetActorLocation(), A.GetActorLocation()) < FVector::DistSquared(Center->GetActorLocation(), B.GetActorLocation());
        });
    }

    if (Result.Num() > K)
    {
        Result.SetNum(K);
    }

    return Result;
}
