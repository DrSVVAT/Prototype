// Copyright Epic Games, Inc. All Rights Reserved.

#include "LearningTrainer.h"

#include "LearningObservation.h"
#include "LearningAction.h"

#include "HAL/Platform.h"
#include "Dom/JsonObject.h"
#include "Misc/Paths.h"

#include "Mac/MacPlatformProcess.h"
#include "Mac/MacPlatform.h"
#include "Apple/ApplePlatformRunnableThread.h"
#include "Containers/UnrealString.h"
#include "Misc/App.h"
#include "Misc/CoreDelegates.h"
#include "Misc/Paths.h"
#include "Misc/StringBuilder.h"
#include "HAL/FileManager.h"
#include "Apple/PreAppleSystemHeaders.h"
#include <mach-o/dyld.h>
#include <mach/thread_act.h>
#include <mach/thread_policy.h>
#include <libproc.h>
#include <spawn.h>
#include "Apple/PostAppleSystemHeaders.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <vector>
#include <string>
#include <spawn.h>
#include <cinttypes>
#include <cstdint>
#if PLATFORM_MAC_X86
    #include <cpuid.h>
#endif

namespace PlatformProcessLimits
{
	enum
	{
		MaxArgvParameters	 = 256
	};
};
// FProcHandle Mooop(const TCHAR* URL, const TCHAR* Parms, bool bLaunchDetached, bool bLaunchHidden, bool bLaunchReallyHidden, uint32* OutProcessID, int32 PriorityModifier, const TCHAR* OptionalWorkingDirectory, void* PipeStdOutChild, void *PipeStdInChild)
// {
// 	SCOPED_AUTORELEASE_POOL;

// 	// @TODO bLaunchHidden bLaunchReallyHidden are not handled
// 	// We need an absolute path to executable
// 	FString ProcessPath = URL;
// 	if (*URL != TEXT('/'))
// 	{
// 		ProcessPath = FPaths::ConvertRelativePathToFull(ProcessPath);
// 	}

// 	// For programs that are wrapped in an App container
// 	{
// 		NSString* nsProcessPath = ProcessPath.GetNSString();
// 		if (![[NSFileManager defaultManager] fileExistsAtPath: nsProcessPath])
// 		{
// 			NSString* AppName = [[nsProcessPath lastPathComponent] stringByDeletingPathExtension];
// 			nsProcessPath = [[[NSWorkspace sharedWorkspace] URLForApplicationWithBundleIdentifier:AppName] path];
// 		}
		
// 		if ([[NSFileManager defaultManager] fileExistsAtPath: nsProcessPath])
// 		{
// 			if([[NSWorkspace sharedWorkspace] isFilePackageAtPath: nsProcessPath])
// 			{
// 				NSBundle* Bundle = [NSBundle bundleWithPath:nsProcessPath];
// 				if(Bundle != nil)
// 				{
// 					nsProcessPath = [Bundle executablePath];
// 					if(nsProcessPath != nil)
// 					{
// 						ProcessPath = nsProcessPath;
// 					}
// 				}
// 			}
// 		}
// 	}

// 	if (!FPaths::FileExists(ProcessPath))
// 	{
// 		return FProcHandle();
// 	}

// 	FString Commandline = FString::Printf(TEXT("\"%s\""), *ProcessPath);
// 	Commandline += TEXT(" ");
// 	Commandline += Parms;

// 	UE_LOG(LogHAL, Verbose, TEXT("FMacPlatformProcess::CreateProc: '%s'"), *Commandline);

// 	TArray<FString> ArgvArray;
// 	int Argc = Commandline.ParseIntoArray(ArgvArray, TEXT(" "), true);
// 	char* Argv[PlatformProcessLimits::MaxArgvParameters + 1] = { NULL };	// last argument is NULL, hence +1
// 	struct CleanupArgvOnExit
// 	{
// 		int Argc;
// 		char** Argv;	// relying on it being long enough to hold Argc elements

// 		CleanupArgvOnExit( int InArgc, char *InArgv[] )
// 			:	Argc(InArgc)
// 			,	Argv(InArgv)
// 		{}

// 		~CleanupArgvOnExit()
// 		{
// 			for (int Idx = 0; Idx < Argc; ++Idx)
// 			{
// 				FMemory::Free(Argv[Idx]);
// 			}
// 		}
// 	} CleanupGuard(Argc, Argv);

// 	// make sure we do not lose arguments with spaces in them due to Commandline.ParseIntoArray breaking them apart above
// 	// @todo this code might need to be optimized somehow and integrated with main argument parser below it
// 	TArray<FString> NewArgvArray;
// 	if (Argc > 0)
// 	{
// 		if (Argc > PlatformProcessLimits::MaxArgvParameters)
// 		{
// 			UE_LOG(LogHAL, Warning, TEXT("FMacPlatformProcess::CreateProc: too many (%d) commandline arguments passed, will only pass %d"),
// 				Argc, PlatformProcessLimits::MaxArgvParameters);
// 			Argc = PlatformProcessLimits::MaxArgvParameters;
// 		}

// 		FString MultiPartArg;
// 		for (int32 Index = 0; Index < Argc; Index++)
// 		{
// 			if (MultiPartArg.IsEmpty())
// 			{
// 				if ((ArgvArray[Index].StartsWith(TEXT("\"")) && !ArgvArray[Index].EndsWith(TEXT("\""))) // check for a starting quote but no ending quote, excludes quoted single arguments
// 					|| (ArgvArray[Index].Contains(TEXT("=\"")) && !ArgvArray[Index].EndsWith(TEXT("\""))) // check for quote after =, but no ending quote, this gets arguments of the type -blah="string string string"
// 					|| ArgvArray[Index].EndsWith(TEXT("=\""))) // check for ending quote after =, this gets arguments of the type -blah=" string string string "
// 				{
// 					MultiPartArg = ArgvArray[Index];
// 				}
// 				else
// 				{
// 					if (ArgvArray[Index].Contains(TEXT("=\"")))
// 					{
// 						FString SingleArg = ArgvArray[Index];
// 						SingleArg = SingleArg.Replace(TEXT("=\""), TEXT("="));
// 						NewArgvArray.Add(SingleArg.TrimQuotes(NULL));
// 					}
// 					else
// 					{
// 						NewArgvArray.Add(ArgvArray[Index].TrimQuotes(NULL));
// 					}
// 				}
// 			}
// 			else
// 			{
// 				MultiPartArg += TEXT(" ");
// 				MultiPartArg += ArgvArray[Index];
// 				if (ArgvArray[Index].EndsWith(TEXT("\"")))
// 				{
// 					if (MultiPartArg.StartsWith(TEXT("\"")))
// 					{
// 						NewArgvArray.Add(MultiPartArg.TrimQuotes(NULL));
// 					}
// 					else if (MultiPartArg.Contains(TEXT("=\"")))
// 					{
// 						FString SingleArg = MultiPartArg.Replace(TEXT("=\""), TEXT("="));
// 						NewArgvArray.Add(SingleArg.TrimQuotes(nullptr));
// 					}
// 					else
// 					{
// 						NewArgvArray.Add(MultiPartArg);
// 					}
// 					MultiPartArg.Empty();
// 				}
// 			}
// 		}
// 	}
// 	// update Argc with the new argument count
// 	Argc = NewArgvArray.Num();

// 	if (Argc > 0)	// almost always, unless there's no program name
// 	{
// 		if (Argc > PlatformProcessLimits::MaxArgvParameters)
// 		{
// 			UE_LOG(LogHAL, Warning, TEXT("FMacPlatformProcess::CreateProc: too many (%d) commandline arguments passed, will only pass %d"),
// 				Argc, PlatformProcessLimits::MaxArgvParameters);
// 			Argc = PlatformProcessLimits::MaxArgvParameters;
// 		}

// 		for (int Idx = 0; Idx < Argc; ++Idx)
// 		{
// 			FTCHARToUTF8 AnsiBuffer(*NewArgvArray[Idx]);
// 			const char* Ansi = AnsiBuffer.Get();
// 			size_t AnsiSize = FCStringAnsi::Strlen(Ansi) + 1;	// will work correctly with UTF-8
// 			check(AnsiSize);

// 			Argv[Idx] = reinterpret_cast< char* >( FMemory::Malloc(AnsiSize) );
// 			check(Argv[Idx]);

// 			FCStringAnsi::Strncpy(Argv[Idx], Ansi, AnsiSize);	// will work correctly with UTF-8
// 		}

// 		// last Argv should be NULL
// 		check(Argc <= PlatformProcessLimits::MaxArgvParameters + 1);
// 		Argv[Argc] = NULL;
// 	}

// 	extern char ** environ;	// provided by libc
// 	pid_t ChildPid = -1;

// 	posix_spawnattr_t SpawnAttr;
// 	posix_spawnattr_init(&SpawnAttr);
// 	short int SpawnFlags = 0;

// 	// Makes spawned processes have its own unique group id so we can kill the entire group without killing the parent
// 	SpawnFlags |= POSIX_SPAWN_SETPGROUP;

// 	// - These are the extra environment keys when turning on GPU Frame Capture via Xcode 11.3 in macOS Catalina (10.15.2):
// 	//		DYLD_INSERT_LIBRARIES, DYMTL_TOOLS_DYLIB_PATH, GPUTOOLS_LOAD_GTMTLCAPTURE, GT_HOST_URL_MTL and METAL_LOAD_INTERPOSER
// 	// - Both DYLD_INSERT_LIBRARIES and METAL_LOAD_INTERPOSER seem to be new for Catalina (10.15.2) compared to Mojave (10.14.6).
// 	// - Using DYLD_INSERT_LIBRARIES seem to be causing a stall at child process startup with Xcode debugger attached to the parent process.
// 	// - Removing DYLD_INSERT_LIBRARIES removes the stall for child process startup which is especially useful for ShaderCompileWorker when invoking MetalCompiler and it's other child processes tools.
// 	char** EnvVariables = environ;
// 	if (WITH_EDITOR && FPlatformMisc::IsDebuggerPresent())
// 	{
// 		int32 NumEnvVariables = 0;
// 		int32 DyldInsertLibrariesEnvVarIndex = -1;

// 		while (environ[NumEnvVariables])
// 		{
// 			if (FCStringAnsi::Strstr(environ[NumEnvVariables], "DYLD_INSERT_LIBRARIES=") == environ[NumEnvVariables])
// 			{
// 				DyldInsertLibrariesEnvVarIndex = NumEnvVariables;
// 			}
// 			++NumEnvVariables;
// 		}

// 		if (DyldInsertLibrariesEnvVarIndex != -1)
// 		{
// 			EnvVariables = (char**)FMemory::Malloc(sizeof(char*) * NumEnvVariables + 1);

// 			int32 NewCount = 0;
// 			for (int32 VarIndex = 0; VarIndex < NumEnvVariables; ++VarIndex)
// 			{
// 				if (VarIndex != DyldInsertLibrariesEnvVarIndex)
// 				{
// 					EnvVariables[NewCount++] = environ[VarIndex];
// 				}
// 			}
// 			EnvVariables[NewCount] = nullptr;
// 		}
// 	}

// 	posix_spawn_file_actions_t FileActions;
// 	posix_spawn_file_actions_init(&FileActions);

// 	if (PipeStdOutChild)
// 	{
// 		posix_spawn_file_actions_adddup2(&FileActions, [(NSFileHandle*)PipeStdOutChild fileDescriptor], STDOUT_FILENO);
// 	}

// 	if (PipeStdInChild)
// 	{
// 		posix_spawn_file_actions_adddup2(&FileActions, [(NSFileHandle*)PipeStdInChild fileDescriptor], STDIN_FILENO);
// 	}

// 	if (OptionalWorkingDirectory)
// 	{
// 		posix_spawn_file_actions_addchdir_np(&FileActions, TCHAR_TO_UTF8(OptionalWorkingDirectory));
// 	}

// 	posix_spawnattr_setflags(&SpawnAttr, SpawnFlags);
// 	UE_LOG(LogLearning, Display, TEXT("PROCESS PATH IS %s"), *ProcessPath);
// 	UE_LOG(LogLearning, Display, TEXT("ARGC IS %d"), Argc);
// 	UE_LOG(LogLearning, Display, TEXT("PARAMS CNT IS %d"), PlatformProcessLimits::MaxArgvParameters + 1);

// 	for (int i = 0; i < Argc; ++i) {

// 		UE_LOG(LogLearning, Display, TEXT("ARGC IS %hs"), Argv[i]);
// 	}

// 	{

// 		pid_t ChildPid = -1;

// 		posix_spawn_file_actions_t FileActions;
// 		posix_spawn_file_actions_init(&FileActions);
// 		const char *OptionalWorkingDirectory = "/Users/a2mogus/free/cars";
// 		posix_spawn_file_actions_addchdir_np(&FileActions, OptionalWorkingDirectory);

// 		int16_t SpawnFlags = 0;
// 		SpawnFlags |= POSIX_SPAWN_SETPGROUP;

// 		posix_spawnattr_t SpawnAttr;
// 		posix_spawnattr_init(&SpawnAttr);
// 		posix_spawnattr_setflags(&SpawnAttr, SpawnFlags);

// 		extern char **environ;    // provided by libc
		
// 		const char * argv[3] = {NULL, NULL, NULL};
// 		argv[0] = "/Users/a2mogus/Documents/Unreal Projects/CarProject/Intermediate/PipInstall/bin/python3";
// 		argv[1] = "/Users/Shared/Epic Games/UE_5.5/Engine/Plugins/Experimental/LearningAgents/Content/Python/train.py";

// 		char * argv2[3] = {NULL, NULL, NULL};
// 		argv2[0] = (char *)argv[0];
// 		argv2[1] = (char *)argv[1];


// 		int Psen = posix_spawn(&ChildPid,
// 										"/Users/a2mogus/Documents/Unreal Projects/CarProject/Intermediate/PipInstall/bin/python3",
// 										&FileActions,
// 										&SpawnAttr,
// 										argv2,
// 										environ);

// 		UE_LOG(LogLearning, Display, TEXT("ARGC IS %d"), Argc);
// 		UE_LOG(LogLearning, Display, TEXT("ARGC IS %d"), Psen);
// 	}

	
// 	int PosixSpawnErrNo = posix_spawn(&ChildPid, TCHAR_TO_UTF8(*ProcessPath), &FileActions, &SpawnAttr, Argv, EnvVariables);
// 	posix_spawn_file_actions_destroy(&FileActions);

// 	posix_spawnattr_destroy(&SpawnAttr);

// 	// Free the allocated memory if we modified the env variables instead of using environ directly
// 	if (EnvVariables != environ)
// 	{
// 		FMemory::Free(EnvVariables);
// 	}

// 	if (PosixSpawnErrNo != 0)
// 	{
// 		UE_LOG(LogHAL, Fatal, TEXT("FMacPlatformProcess::CreateProc: posix_spawn() failed (%d, %s)"), PosixSpawnErrNo, UTF8_TO_TCHAR(strerror(PosixSpawnErrNo)));
// 		return FProcHandle();	// produce knowingly invalid handle if for some reason Fatal log (above) returns
// 	}

// 	if (PriorityModifier != 0)
// 	{
// 		PriorityModifier = MIN(PriorityModifier, -2);
// 		PriorityModifier = MAX(PriorityModifier, 2);
// 		// priority values: 20 = lowest, 10 = low, 0 = normal, -10 = high, -20 = highest
// 		setpriority(PRIO_PROCESS, ChildPid, -PriorityModifier * 10);
// 	}

// 	if (OutProcessID)
// 	{
// 		*OutProcessID = ChildPid;
// 	}

// 	// [RCL] 2015-03-11 @FIXME: is bLaunchDetached usable when determining whether we're in 'fire and forget' mode? This doesn't exactly match what bLaunchDetached is used for.
// 	return FProcHandle();
// 	}


namespace UE::Learning
{

	FSubprocess::~FSubprocess()
	{
		Terminate();
	}

	bool FSubprocess::Launch(const FString& Path, const FString& Params, const ESubprocessFlags Flags)
	{
		FString Root("/Users/a2mogus/free/cars");
		UE_LOG(LogLearning, Display, TEXT("MOP V STOYLO!!!!!!!!"))
		UE_LOG(LogLearning, Display, TEXT("PATH: %s"), *Path)
		UE_LOG(LogLearning, Display, TEXT("PARAMS: %s"), *Params)
		UE_LOG(LogLearning, Display, TEXT("FLAGS: %d"), Flags)
		// UE_LOG(LogLearning, Display, TEXT("ROOT: %s"), *FPaths::RootDir())
		UE_LOG(LogLearning, Display, TEXT("ROOT: %s"), *Root)
		

		ensureMsgf(!bIsLaunched, TEXT("Subprocess already launched."));

		Terminate();

		const bool bCreatePipes = !(Flags & ESubprocessFlags::NoRedirectOutput);
		const bool bHideWindow = !(Flags & ESubprocessFlags::ShowWindow);

		if (bCreatePipes && !FPlatformProcess::CreatePipe(ReadPipe, WritePipe))
		{
			return false;
		}
		unsigned int z = 0;

		ProcessHandle = FPlatformProcess::CreateProc(*Path, *Params, false, bHideWindow, bHideWindow, &z, 0, *FPaths::RootDir(), WritePipe, ReadPipe);
		bIsLaunched = true;
		return true;
	}

	bool FSubprocess::IsRunning() const
	{
		return bIsLaunched && FPlatformProcess::IsProcRunning(const_cast<FProcHandle&>(ProcessHandle));
	}

	void FSubprocess::Terminate()
	{
		if (IsRunning())
		{
			UE_LOG(LogLearning, Display, TEXT("Terminating Subprocess..."));

			FPlatformProcess::TerminateProc(ProcessHandle, true);
		}

		Update();
	}

	bool FSubprocess::Update()
	{
		// Do nothing if the process is not launched
		if (!bIsLaunched)
		{
			return false;
		}

		// Append the process stdout to the buffer
		OutputBuffer += FPlatformProcess::ReadPipe(ReadPipe);

		// Output all the complete lines
		int32 LineStartIdx = 0;
		for (int32 Idx = 0; Idx < OutputBuffer.Len(); Idx++)
		{
			if (OutputBuffer[Idx] == '\r' || OutputBuffer[Idx] == '\n')
			{
				UE_LOG(LogLearning, Display, TEXT("Subprocess: %s"), *OutputBuffer.Mid(LineStartIdx, Idx - LineStartIdx));

				if (OutputBuffer[Idx] == '\r' && Idx + 1 < OutputBuffer.Len() && OutputBuffer[Idx + 1] == '\n')
				{
					Idx++;
				}

				LineStartIdx = Idx + 1;
			}
		}

		// Remove all the complete lines from the buffer
		OutputBuffer.MidInline(LineStartIdx, MAX_int32, EAllowShrinking::Yes);

		// If the process is no longer running then close the pipes
		if (!IsRunning())
		{
			FPlatformProcess::ClosePipe(ReadPipe, WritePipe);
			ReadPipe = nullptr;
			WritePipe = nullptr;
			bIsLaunched = false;
			return false;
		}
		
		return true;
	}

}

namespace UE::Learning::Trainer
{

	TSharedPtr<FJsonObject> ConvertObservationSchemaToJSON(
		const Observation::FSchema& ObservationSchema,
		const Observation::FSchemaElement& ObservationSchemaElement)
	{
		TSharedPtr<FJsonObject> Object = MakeShared<FJsonObject>();
		Object->SetNumberField(TEXT("VectorSize"), ObservationSchema.GetObservationVectorSize(ObservationSchemaElement));
		Object->SetNumberField(TEXT("EncodedSize"), ObservationSchema.GetEncodedVectorSize(ObservationSchemaElement));

		switch (ObservationSchema.GetType(ObservationSchemaElement))
		{
		case Observation::EType::Null:
		{
			Object->SetStringField(TEXT("Type"), TEXT("Null"));
			break;
		}

		case Observation::EType::Continuous:
		{
			const Observation::FSchemaContinuousParameters Parameters = ObservationSchema.GetContinuous(ObservationSchemaElement);
			Object->SetStringField(TEXT("Type"), TEXT("Continuous"));
			Object->SetNumberField(TEXT("Num"), Parameters.Num);
			break;
		}

		case Observation::EType::And:
		{
			const Observation::FSchemaAndParameters Parameters = ObservationSchema.GetAnd(ObservationSchemaElement);
			
			Object->SetStringField(TEXT("Type"), TEXT("And"));

			TSharedPtr<FJsonObject> SubObject = MakeShared<FJsonObject>();
			for (int32 SubElementIdx = 0; SubElementIdx < Parameters.Elements.Num(); SubElementIdx++)
			{
				TSharedPtr<FJsonObject> SubElement = ConvertObservationSchemaToJSON(ObservationSchema, Parameters.Elements[SubElementIdx]);
				SubElement->SetNumberField(TEXT("Index"), SubElementIdx);
				SubObject->SetObjectField(Parameters.ElementNames[SubElementIdx].ToString(), SubElement);
			}

			Object->SetObjectField(TEXT("Elements"), SubObject);
			break;
		}

		case Observation::EType::OrExclusive:
		{
			const Observation::FSchemaOrExclusiveParameters Parameters = ObservationSchema.GetOrExclusive(ObservationSchemaElement);

			Object->SetStringField(TEXT("Type"), TEXT("OrExclusive"));
			Object->SetNumberField(TEXT("EncodingSize"), Parameters.EncodingSize);

			TSharedPtr<FJsonObject> SubObject = MakeShared<FJsonObject>();
			for (int32 SubElementIdx = 0; SubElementIdx < Parameters.Elements.Num(); SubElementIdx++)
			{
				TSharedPtr<FJsonObject> SubElement = ConvertObservationSchemaToJSON(ObservationSchema, Parameters.Elements[SubElementIdx]);
				SubElement->SetNumberField(TEXT("Index"), SubElementIdx);
				SubObject->SetObjectField(Parameters.ElementNames[SubElementIdx].ToString(), SubElement);
			}

			Object->SetObjectField(TEXT("Elements"), SubObject);
			break;
		}

		case Observation::EType::OrInclusive:
		{
			const Observation::FSchemaOrInclusiveParameters Parameters = ObservationSchema.GetOrInclusive(ObservationSchemaElement);

			Object->SetStringField(TEXT("Type"), TEXT("OrInclusive"));
			Object->SetNumberField(TEXT("AttentionEncodingSize"), Parameters.AttentionEncodingSize);
			Object->SetNumberField(TEXT("AttentionHeadNum"), Parameters.AttentionHeadNum);
			Object->SetNumberField(TEXT("ValueEncodingSize"), Parameters.ValueEncodingSize);

			TSharedPtr<FJsonObject> SubObject = MakeShared<FJsonObject>();
			for (int32 SubElementIdx = 0; SubElementIdx < Parameters.Elements.Num(); SubElementIdx++)
			{
				TSharedPtr<FJsonObject> SubElement = ConvertObservationSchemaToJSON(ObservationSchema, Parameters.Elements[SubElementIdx]);
				SubElement->SetNumberField(TEXT("Index"), SubElementIdx);
				SubObject->SetObjectField(Parameters.ElementNames[SubElementIdx].ToString(), SubElement);
			}

			Object->SetObjectField(TEXT("Elements"), SubObject);
			break;
		}

		case Observation::EType::Array:
		{
			const Observation::FSchemaArrayParameters Parameters = ObservationSchema.GetArray(ObservationSchemaElement);

			Object->SetStringField(TEXT("Type"), TEXT("Array"));
			Object->SetNumberField(TEXT("Num"), Parameters.Num);
			Object->SetObjectField(TEXT("Element"), ConvertObservationSchemaToJSON(ObservationSchema, Parameters.Element));
			break;
		}

		case Observation::EType::Set:
		{
			const Observation::FSchemaSetParameters Parameters = ObservationSchema.GetSet(ObservationSchemaElement);

			Object->SetStringField(TEXT("Type"), TEXT("Set"));
			Object->SetNumberField(TEXT("MaxNum"), Parameters.MaxNum);
			Object->SetNumberField(TEXT("AttentionEncodingSize"), Parameters.AttentionEncodingSize);
			Object->SetNumberField(TEXT("AttentionHeadNum"), Parameters.AttentionHeadNum);
			Object->SetNumberField(TEXT("ValueEncodingSize"), Parameters.ValueEncodingSize);
			Object->SetObjectField(TEXT("Element"), ConvertObservationSchemaToJSON(ObservationSchema, Parameters.Element));
			break;
		}

		case Observation::EType::Encoding:
		{
			const Observation::FSchemaEncodingParameters Parameters = ObservationSchema.GetEncoding(ObservationSchemaElement);

			Object->SetStringField(TEXT("Type"), TEXT("Encoding"));
			Object->SetNumberField(TEXT("EncodingSize"), Parameters.EncodingSize);
			Object->SetObjectField(TEXT("Element"), ConvertObservationSchemaToJSON(ObservationSchema, Parameters.Element));
			break;
		}

		default:
			UE_LEARNING_NOT_IMPLEMENTED();
		}

		return Object;
	}

	TSharedPtr<FJsonObject> ConvertActionSchemaToJSON(
		const Action::FSchema& ActionSchema,
		const Action::FSchemaElement& ActionSchemaElement)
	{
		TSharedPtr<FJsonObject> Object = MakeShared<FJsonObject>();
		Object->SetNumberField(TEXT("VectorSize"), ActionSchema.GetActionVectorSize(ActionSchemaElement));
		Object->SetNumberField(TEXT("DistributionSize"), ActionSchema.GetActionDistributionVectorSize(ActionSchemaElement));
		Object->SetNumberField(TEXT("EncodedSize"), ActionSchema.GetEncodedVectorSize(ActionSchemaElement));

		switch (ActionSchema.GetType(ActionSchemaElement))
		{
		case Action::EType::Null:
		{
			Object->SetStringField(TEXT("Type"), TEXT("Null"));
			break;
		}

		case Action::EType::Continuous:
		{
			const Action::FSchemaContinuousParameters Parameters = ActionSchema.GetContinuous(ActionSchemaElement);

			Object->SetStringField(TEXT("Type"), TEXT("Continuous"));
			Object->SetNumberField(TEXT("Num"), Parameters.Num);
			break;
		}

		case Action::EType::DiscreteExclusive:
		{
			const Action::FSchemaDiscreteExclusiveParameters Parameters = ActionSchema.GetDiscreteExclusive(ActionSchemaElement);

			Object->SetStringField(TEXT("Type"), TEXT("DiscreteExclusive"));
			Object->SetNumberField(TEXT("Num"), Parameters.Num);
			break;
		}

		case Action::EType::DiscreteInclusive:
		{
			const Action::FSchemaDiscreteInclusiveParameters Parameters = ActionSchema.GetDiscreteInclusive(ActionSchemaElement);

			Object->SetStringField(TEXT("Type"), TEXT("DiscreteInclusive"));
			Object->SetNumberField(TEXT("Num"), Parameters.Num);
			break;
		}

		case Action::EType::And:
		{
			const Action::FSchemaAndParameters Parameters = ActionSchema.GetAnd(ActionSchemaElement);

			Object->SetStringField(TEXT("Type"), TEXT("And"));

			TSharedPtr<FJsonObject> SubObject = MakeShared<FJsonObject>();
			for (int32 SubElementIdx = 0; SubElementIdx < Parameters.Elements.Num(); SubElementIdx++)
			{
				TSharedPtr<FJsonObject> SubElement = ConvertActionSchemaToJSON(ActionSchema, Parameters.Elements[SubElementIdx]);
				SubElement->SetNumberField(TEXT("Index"), SubElementIdx);
				SubObject->SetObjectField(Parameters.ElementNames[SubElementIdx].ToString(), SubElement);
			}

			Object->SetObjectField(TEXT("Elements"), SubObject);
			break;
		}

		case Action::EType::OrExclusive:
		{
			const Action::FSchemaOrExclusiveParameters Parameters = ActionSchema.GetOrExclusive(ActionSchemaElement);

			Object->SetStringField(TEXT("Type"), TEXT("OrExclusive"));

			TSharedPtr<FJsonObject> SubObject = MakeShared<FJsonObject>();
			for (int32 SubElementIdx = 0; SubElementIdx < Parameters.Elements.Num(); SubElementIdx++)
			{
				TSharedPtr<FJsonObject> SubElement = ConvertActionSchemaToJSON(ActionSchema, Parameters.Elements[SubElementIdx]);
				SubElement->SetNumberField(TEXT("Index"), SubElementIdx);
				SubObject->SetObjectField(Parameters.ElementNames[SubElementIdx].ToString(), SubElement);
			}

			Object->SetObjectField(TEXT("Elements"), SubObject);
			break;
		}

		case Action::EType::OrInclusive:
		{
			const Action::FSchemaOrInclusiveParameters Parameters = ActionSchema.GetOrInclusive(ActionSchemaElement);

			Object->SetStringField(TEXT("Type"), TEXT("OrInclusive"));

			TSharedPtr<FJsonObject> SubObject = MakeShared<FJsonObject>();
			for (int32 SubElementIdx = 0; SubElementIdx < Parameters.Elements.Num(); SubElementIdx++)
			{
				TSharedPtr<FJsonObject> SubElement = ConvertActionSchemaToJSON(ActionSchema, Parameters.Elements[SubElementIdx]);
				SubElement->SetNumberField(TEXT("Index"), SubElementIdx);
				SubObject->SetObjectField(Parameters.ElementNames[SubElementIdx].ToString(), SubElement);
			}

			Object->SetObjectField(TEXT("Elements"), SubObject);
			break;
		}

		case Action::EType::Array:
		{
			const Action::FSchemaArrayParameters Parameters = ActionSchema.GetArray(ActionSchemaElement);

			Object->SetStringField(TEXT("Type"), TEXT("Array"));
			Object->SetNumberField(TEXT("Num"), Parameters.Num);
			Object->SetObjectField(TEXT("Element"), ConvertActionSchemaToJSON(ActionSchema, Parameters.Element));
			break;
		}

		case Action::EType::Encoding:
		{
			const Action::FSchemaEncodingParameters Parameters = ActionSchema.GetEncoding(ActionSchemaElement);

			Object->SetStringField(TEXT("Type"), TEXT("Encoding"));
			Object->SetNumberField(TEXT("EncodingSize"), Parameters.EncodingSize);
			Object->SetObjectField(TEXT("Element"), ConvertActionSchemaToJSON(ActionSchema, Parameters.Element));
			break;
		}

		default:
			UE_LEARNING_NOT_IMPLEMENTED();
		}

		return Object;
	}

	const TCHAR* GetDeviceString(const ETrainerDevice Device)
	{
		switch (Device)
		{
		case ETrainerDevice::GPU: return TEXT("GPU");
		case ETrainerDevice::CPU: return TEXT("CPU");
		default: UE_LEARNING_NOT_IMPLEMENTED(); return TEXT("Unknown");
		}
	}

	const TCHAR* GetResponseString(const ETrainerResponse Response)
	{
		switch (Response)
		{
		case ETrainerResponse::Success: return TEXT("Success");
		case ETrainerResponse::Unexpected: return TEXT("Unexpected communication received");
		case ETrainerResponse::Completed: return TEXT("Training completed");
		case ETrainerResponse::Stopped: return TEXT("Training stopped");
		case ETrainerResponse::Timeout: return TEXT("Communication timeout");
		default: UE_LEARNING_NOT_IMPLEMENTED(); return TEXT("Unknown");
		}
	}

	float DiscountFactorFromHalfLife(const float HalfLife, const float DeltaTime)
	{
		return FMath::Pow(0.5f, DeltaTime / FMath::Max(HalfLife, UE_SMALL_NUMBER));
	}

	float DiscountFactorFromHalfLifeSteps(const int32 HalfLifeSteps)
	{
		UE_LEARNING_CHECKF(HalfLifeSteps >= 1, TEXT("Number of HalfLifeSteps should be at least 1 but got %i"), HalfLifeSteps);

		return FMath::Pow(0.5f, 1.0f / FMath::Max(HalfLifeSteps, 1));
	}

	FString GetPythonExecutablePath(const FString& IntermediateDir)
	{
		UE_LEARNING_CHECKF(PLATFORM_WINDOWS || PLATFORM_MAC || PLATFORM_LINUX, TEXT("Python only supported on Windows, Mac, and Linux."));

		return IntermediateDir / TEXT("PipInstall") / (PLATFORM_WINDOWS ? TEXT("Scripts/python.exe") : TEXT("bin/python3"));
	}

	FString GetSitePackagesPath(const FString& EngineDir)
	{
		UE_LEARNING_CHECKF(PLATFORM_WINDOWS || PLATFORM_MAC || PLATFORM_LINUX, TEXT("Python only supported on Windows, Mac, and Linux."));

		return EngineDir / TEXT("Plugins/Experimental/PythonFoundationPackages/Content/Python/Lib") / FPlatformMisc::GetUBTPlatform() / TEXT("site-packages");
	}

	FString GetPythonContentPath(const FString& EngineDir)
	{
		return EngineDir / TEXT("Plugins/Experimental/LearningAgents/Content/Python/");
	}

	FString GetProjectPythonContentPath()
	{
		return FPaths::ProjectContentDir() / TEXT("Python/");
	}

	FString GetIntermediatePath(const FString& IntermediateDir)
	{
		return IntermediateDir / TEXT("LearningAgents");
	}

}