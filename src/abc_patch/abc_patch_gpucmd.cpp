#include <stdexcept>
#include <iostream>
#include <string>
#include "CLI11.hpp"

#include "abc_patch/abc_patch_int.h"
#include "misc/string_utils.h"

using strUtil::descWithDefault, strUtil::concat;

namespace abcPatch {

std::string restoreCommandString(int argc, char ** argv) {
    std::string cmd = "";
    for (int i = 0; i < argc; i++) {
        cmd += argv[i];
        if (i != argc - 1)
            cmd += " ";
    }
    return cmd;
}

bool parseOptions(CLI::App & parser, int argc, char ** argv) {
    try {
        parser.parse(restoreCommandString(argc, argv), true);
    } catch (const CLI::CallForHelp &e) {
        std::cout << parser.help();
        return false;
    } catch (const CLI::ParseError &e) {
        std::cout << parser.help();
        return false;
    }
    return true;
}

bool checkGpuManState() {
    if (!gpuManIsActive()) {
        std::cout << "GPU engine is not in the active state!\n";
        return false;
    }
    return true;
}


// ABC command handlers

int Abc_CommandGpuRead(Abc_Frame_t * pAbc, int argc, char ** argv) {
    std::string path = "";

    CLI::App parser("Load AIG file into the GPU engine");
    parser.add_option("file_path", path, 
                      "path to the file to be read")->required();

    if (!parseOptions(parser, argc, argv))
        return 1;

    int ret = getGpuMan()->readFile(path.c_str());
    if (ret)
        gpuManSetActive();
    else
        gpuManSetInactive();

    return 1 - ret;
}

int Abc_CommandGpuWrite(Abc_Frame_t * pAbc, int argc, char ** argv) {
    std::string path = "";

    CLI::App parser("Dump the AIG in the GPU engine into a file");
    parser.add_option("file_path", path, 
                      "path to the file to be written")->required();

    if (!parseOptions(parser, argc, argv) || !checkGpuManState())
        return 1;

    getGpuMan()->saveFile(path.c_str());
    return 0;
}

int Abc_CommandGpuTime(Abc_Frame_t * pAbc, int argc, char ** argv) {
    if (!checkGpuManState())
        return 1;

    getGpuMan()->printTime();
    return 0;
}

int Abc_CommandGpuPrintStats(Abc_Frame_t * pAbc, int argc, char ** argv) {
    if (!checkGpuManState())
        return 1;

    getGpuMan()->printStats();
    return 0;
}

int Abc_CommandGpuBalance(Abc_Frame_t * pAbc, int argc, char ** argv) {
    bool fSortWithoutDecId = false;

    CLI::App parser("Perform GPU AIG balancing");
    parser.add_flag("-s", fSortWithoutDecId, 
        "if provided, not using id as tie-breaker when combining same delay nodes");
    
    if (!parseOptions(parser, argc, argv) || !checkGpuManState())
        return 1;
    
    getGpuMan()->balance(fSortWithoutDecId ? 0 : 1);
    return 0;
}

int Abc_CommandGpuRewrite(Abc_Frame_t * pAbc, int argc, char ** argv) {
    bool fUseZeros = false, fGPUDeduplicate = false;

    CLI::App parser("Perform GPU AIG rewriting (DAC'22)");
    parser.add_flag("-z", fUseZeros, "if provided, allow zero gain replacement");
    parser.add_flag("-d", fGPUDeduplicate, 
        concat({
            "toggle GPU/CPU strash and dangling node cleanup in the end [default = ",
            fGPUDeduplicate ? "GPU" : "CPU",
            "]"
        }));
    
    if (!parseOptions(parser, argc, argv) || !checkGpuManState())
        return 1;
    
    getGpuMan()->rewrite(fUseZeros, fGPUDeduplicate);
    return 0;
}

int Abc_CommandGpuRefactor(Abc_Frame_t * pAbc, int argc, char ** argv) {
    bool fUseZeros = false, fAlgMFFC = false;
    int cutSize = 12;

    CLI::App parser("Perform GPU AIG refactoring");
    parser.add_flag("-z", fUseZeros, "if provided, allow zero gain replacement");
    parser.add_flag("-m", fAlgMFFC, 
        concat({
            "toggle GPU-rewrite-like (DAC'22)/MFFC-based (DAC'23) algorithm for refactoring",
            " [default = ",
            fAlgMFFC ? "MFFC-based" : "GPU-rewrite-like",
            "]"
        }));
    parser.add_option("-K", cutSize, descWithDefault("maximum cut size", cutSize));

    if (!parseOptions(parser, argc, argv) || !checkGpuManState())
        return 1;
    
    getGpuMan()->refactor(fAlgMFFC, fUseZeros, cutSize);
    return 0;
}

int Abc_CommandGpuStrash(Abc_Frame_t * pAbc, int argc, char ** argv) {
    bool fCPU = false;

    CLI::App parser("Perform GPU AIG strashing and dangling node cleanup");
    parser.add_flag("-c", fCPU, 
        concat({
            "toggle GPU/CPU strash and dangling node cleanup [default = ",
            fCPU ? "CPU" : "GPU",
            "]"
        }));
    
    if (!parseOptions(parser, argc, argv) || !checkGpuManState())
        return 1;
    
    getGpuMan()->strash(fCPU, true);
    return 0;
}

int Abc_CommandGpuResyn2(Abc_Frame_t * pAbc, int argc, char ** argv) {
    // command:
    // b; rw -d; rf -m; st; b; rw -d; rw -z -d; rw -z -d; b -s; rf -m -z; st; 
    // rw -z -d; rw -z -d; b -s

    int cutSize = 12;

    CLI::App parser("Perform GPU resyn2");
    parser.add_option("-K", cutSize, 
        descWithDefault("maximum cut size used in refactoring", cutSize));
    
    if (!parseOptions(parser, argc, argv) || !checkGpuManState())
        return 1;
    
    AIGMan * pMan = getGpuMan();
    
    pMan->balance(1);
    pMan->rewrite(false, true);
    pMan->refactor(true, false, cutSize);
    pMan->strash(false, true);
    pMan->balance(1);
    pMan->rewrite(false, true);
    pMan->rewrite(true, true);
    pMan->rewrite(true, true);
    pMan->balance(0);
    pMan->refactor(true, true, cutSize);
    pMan->strash(false, true);
    pMan->rewrite(true, true);
    pMan->rewrite(true, true);
    pMan->balance(0);

    return 0;
}

int Abc_CommandGpuGet(Abc_Frame_t * pAbc, int argc, char ** argv) {
    Abc_Ntk_t * pNtk = Abc_FrameReadNtk(pAbc);
    AIGMan * pMan = getGpuMan();

    bool ret = AbcNtkToGpuMan(pNtk, pMan);
    if (ret) {
        gpuManSetActive();
        pMan->setAigCreated(1);
    } else {
        gpuManSetInactive();
        pMan->setAigCreated(0);
    }
    pMan->setPrevCmdRewrite(0);
    
    return ret ? 0 : 1;
}

int Abc_CommandGpuPut(Abc_Frame_t * pAbc, int argc, char ** argv) {
    if (!checkGpuManState())
        return 1;
    
    AIGMan * pMan = getGpuMan();
    Abc_Ntk_t * pNtkNew = GpuManToAbcNtk(pMan);
    Abc_FrameReplaceCurrentNetwork(pAbc, pNtkNew);

    gpuManSetInactive();
    pMan->setAigCreated(0);
    pMan->setPrevCmdRewrite(0);

    return 0;
}


// register all handlers to ABC
void registerAllAbcCommands(Abc_Frame_t * pAbc) {
    Cmd_CommandAdd(pAbc, "GPU", "gread", Abc_CommandGpuRead, 0);
    Cmd_CommandAdd(pAbc, "GPU", "gwrite", Abc_CommandGpuWrite, 0);
    Cmd_CommandAdd(pAbc, "GPU", "gtime", Abc_CommandGpuTime, 0);
    Cmd_CommandAdd(pAbc, "GPU", "gps", Abc_CommandGpuPrintStats, 0);

    Cmd_CommandAdd(pAbc, "GPU", "gb", Abc_CommandGpuBalance, 0);
    Cmd_CommandAdd(pAbc, "GPU", "grw", Abc_CommandGpuRewrite, 0);
    Cmd_CommandAdd(pAbc, "GPU", "grf", Abc_CommandGpuRefactor, 0);
    Cmd_CommandAdd(pAbc, "GPU", "gst", Abc_CommandGpuStrash, 0);
    Cmd_CommandAdd(pAbc, "GPU", "gresyn2", Abc_CommandGpuResyn2, 0);

    Cmd_CommandAdd(pAbc, "GPU", "gget", Abc_CommandGpuGet, 0);
    Cmd_CommandAdd(pAbc, "GPU", "gput", Abc_CommandGpuPut, 0);
}

} // namespace abcPatch
