#include <stdexcept>
#include <iostream>
#include <sstream>
#include <string>
#include "CLI11.hpp"
#include "command_manager.h"
#include "misc/string_utils.h"

using strUtil::descWithDefault;

int parseCmd(CLI::App & parser, const std::vector<std::string> & vLiterals){
    std::string cmd = "";
    for (int i = 0; i < vLiterals.size(); i++) {
        cmd += vLiterals[i];
        if (i != vLiterals.size() - 1)
            cmd += " ";
    }

    try {
        parser.parse(cmd, true);
    } catch (const CLI::CallForHelp &e) {
        std::cout << parser.help();
        return 0;
    } catch (const CLI::ParseError &e) {
        return 1;
    }
    return 2;
}

// *************** command handler implementations ***************

int readHandler(AIGMan & aigman, const std::vector<std::string> & vLiterals) {
    if (vLiterals.size() < 2)
        return 1;
    aigman.readFile(vLiterals[1].c_str());
    return 0;
}

int writeHandler(AIGMan & aigman, const std::vector<std::string> & vLiterals) {
    if (vLiterals.size() < 2)
        return 1;

    if (!strUtil::endsWith(vLiterals[1], ".aig")) {
        printf("write: only support .aig format!\n");
        return 1;
    }
    aigman.saveFile(vLiterals[1].c_str());
    return 0;
}

int timeHandler(AIGMan & aigman, const std::vector<std::string> & vLiterals) {
    aigman.printTime();
    return 0;
}

int printStatsHandler(AIGMan & aigman, const std::vector<std::string> & vLiterals) {
    aigman.printStats();
    return 0;
}

int balanceHandler(AIGMan & aigman, const std::vector<std::string> & vLiterals) {
    bool fSort = false;
    CLI::App parser("Perform Balance");
    parser.add_flag("-s", fSort, descWithDefault("using id when sorting", fSort));
    parser.add_option("-v", aigman.verbose, descWithDefault("verbose level", aigman.verbose));

    int ret;
    if((ret = parseCmd(parser, vLiterals)) < 2 )
        return ret;

    int sortDecId = 1;
    if(fSort){
        sortDecId = 0;
        printf("** balance without using id as tie break when sorting.\n");
    }

    aigman.balance(sortDecId);
    return 0;
}

int rewriteHandler(AIGMan & aigman, const std::vector<std::string> & vLiterals) {
    bool fUseZeros = false, fGPUDeduplicate = true;

    CLI::App parser("Perform rewrite");
    parser.add_flag("-z", fUseZeros, descWithDefault("using zero", fUseZeros));
    parser.add_flag("-d", fGPUDeduplicate, descWithDefault("using GPU deduplicate", fGPUDeduplicate));
    parser.add_option("-v", aigman.verbose, descWithDefault("verbose level", aigman.verbose));

    int ret;
    if((ret = parseCmd(parser, vLiterals)) < 2 )
        return ret;

    aigman.rewrite(fUseZeros, fGPUDeduplicate);
    return 0;
}

int refactorHandler(AIGMan & aigman, const std::vector<std::string> & vLiterals) {
    bool fUseZeros = false, fAlgMFFC = false;
    int cutSize = 12;

    CLI::App parser("Perform refactor");
    parser.add_flag("-z", fUseZeros, descWithDefault("fUseZeros", fUseZeros));
    parser.add_flag("-m", fAlgMFFC, descWithDefault("fAlgMFFC", fAlgMFFC));
    parser.add_option("-K", cutSize, descWithDefault("cut size (<=15)", cutSize));
    parser.add_option("-v", aigman.verbose, descWithDefault("verbose level", aigman.verbose));

    int ret;
    if((ret = parseCmd(parser, vLiterals)) < 2 )
        return ret;

    aigman.refactor(fAlgMFFC, fUseZeros, cutSize);
    return 0;
}

int strashHandler(AIGMan & aigman, const std::vector<std::string> & vLiterals) {
    bool fCPU = false;

    CLI::App parser("Perform Structural Hash");
    parser.add_flag("-c", fCPU, descWithDefault("fCPU", fCPU));
    parser.add_option("-v", aigman.verbose, descWithDefault("verbose level", aigman.verbose));

    int ret;
    if((ret = parseCmd(parser, vLiterals)) < 2 )
        return ret;

    aigman.strash(fCPU, true);
    return 0;
}

int resyn2Handler(AIGMan & aigman, const std::vector<std::string> & vLiterals) {
    // command:
    // b; rw -d; rf -m; st; b; rw -d; rw -z -d; rw -z -d; b -s; rf -m -z; st; 
    // rw -z -d; rw -z -d; b -s

    int cutSize = 12;

    CLI::App parser("Perform resyn2");
    parser.add_option("-K", cutSize, 
                      descWithDefault("maximum cut size used in refactoring", cutSize));
    parser.add_option("-v", aigman.verbose, descWithDefault("verbose level", aigman.verbose));
    
    int ret;
    if((ret = parseCmd(parser, vLiterals)) < 2 )
        return ret;

    if (aigman.nNodes == 0)
        return 0;

clock_t startTime = clock();

    aigman.balance(1);
    aigman.rewrite(false, true);
    aigman.refactor(true, false, cutSize);
    aigman.strash(false, true);
    aigman.balance(1);
    aigman.rewrite(false, true);
    aigman.rewrite(true, true);
    aigman.rewrite(true, true);
    aigman.balance(0);
    aigman.refactor(true, true, cutSize);
    aigman.strash(false, true);
    aigman.rewrite(true, true);
    aigman.rewrite(true, true);
    aigman.balance(0);
    aigman.strash(false, true);

aigman.setAlgTime(startTime);
aigman.setFullTime(startTime);

    return 0;
}

int resubHandler(AIGMan & aigman, const std::vector<std::string> & vLiterals) {
    int cutSize = 8,  addNodes = 1, verbose=1;
    bool fUseZeros = false, fUseConstr = true, fUpdateLevel = false;

    CLI::App parser("Perform recorded library based optimization");
    parser.add_flag("-z", fUseZeros, descWithDefault("fUseZeros", fUseZeros));
    parser.add_flag("-l", fUpdateLevel, descWithDefault("fUpdateLevel", fUpdateLevel));
    parser.add_option("-k", cutSize, descWithDefault("cut size (<=15)", cutSize));
    parser.add_option("-n", addNodes, descWithDefault("max number of nodes to add (0 <= num <=3 )", addNodes));
    parser.add_option("-v", aigman.verbose, descWithDefault("verbose level", aigman.verbose));

    int ret;
    if((ret = parseCmd(parser, vLiterals)) < 2 )
        return ret;

    assert(cutSize>0);  assert(cutSize<=15);
    assert(addNodes>=0);    assert(addNodes<=3);

    aigman.resub(fUseZeros, fUseConstr, fUpdateLevel, cutSize, addNodes);
    return 0;
}

int resyn2rsHandler(AIGMan & aigman, const std::vector<std::string> & vLiterals) {
    // command:
    // b; st; rs -k 6; rw -d; rs -k 6 -n 2; st; rf -m -K 10; st;
    // rs -k 8; b; st; rs -k 8 -n 2; rw -d;
    // rs -k 10; rw -z -d; rs -k 10 -n 2; b; st;
    // rs -k 10; st; rf -m -z -K 10; st; rs -k 10 -n 2; rw -d -z; b; st;

    int cutSize = 12;
    CLI::App parser("Perform resyn2rs");
    parser.add_option("-K", cutSize, 
                      descWithDefault("maximum cut size used in refactoring", cutSize));
    parser.add_option("-v", aigman.verbose, descWithDefault("verbose level", aigman.verbose));
    
    int ret;
    if((ret = parseCmd(parser, vLiterals)) < 2 )
        return ret;

    if (aigman.nNodes == 0)
        return 0;

clock_t startTime = clock();

    aigman.balance(1);
    aigman.strash(false, true);
    aigman.resub(false, true, false, 6, 1);
    aigman.rewrite(false, true);
    aigman.resub(false, true, false, 6, 2);
    aigman.strash(false, true);
    aigman.refactor(true, false, cutSize);
    aigman.strash(false, true);

    aigman.resub(false, true, false, 8, 1);
    aigman.balance(1);
    aigman.strash(false, true);
    aigman.resub(false, true, false, 8, 2);
    aigman.rewrite(false, true);

    aigman.resub(false, true, false, 10, 1);
    aigman.rewrite(true, true);
    aigman.resub(false, true, false, 10, 2);
    aigman.balance(1);
    aigman.strash(false, true);

    aigman.resub(false, true, false, 10, 1);
    aigman.strash(false, true);
    aigman.refactor(true, true, cutSize);
    aigman.strash(false, true);
    aigman.resub(false, true, false, 10, 2);
    aigman.rewrite(true, true);
    aigman.balance(1);
    aigman.strash(false, true);

aigman.setAlgTime(startTime);
aigman.setFullTime(startTime);

    return 0;
}

// add an register entry here when add a new command
void CmdMan::registerAllCommands() {
    // basic commands
    registerCommand("read", readHandler);
    registerCommand("write", writeHandler);
    registerCommand("time", timeHandler);
    registerCommand("ps", printStatsHandler);
    registerCommand("print_stats", printStatsHandler);

    // main algorithms
    registerCommand("b", balanceHandler);
    registerCommand("balance", balanceHandler);
    registerCommand("rw", rewriteHandler);
    registerCommand("rewrite", rewriteHandler);
    registerCommand("rf", refactorHandler);
    registerCommand("refactor", refactorHandler);
    registerCommand("rs", resubHandler);
    registerCommand("resub", resubHandler);
    registerCommand("st", strashHandler);
    registerCommand("strash", strashHandler);

    registerCommand("resyn2", resyn2Handler);
    registerCommand("resyn2rs", resyn2rsHandler);
    registerCommand("r2rs", resyn2rsHandler);
}

void CmdMan::registerCommand(const std::string & cmd, const CommandHandler & cmdHandlerFunc) {
    if (htCmdFuncs.find(cmd) != htCmdFuncs.end()) {
        printf("Command %s already added!\n", cmd.c_str());
        return;
    }

    htCmdFuncs[cmd] = cmdHandlerFunc;
}

void CmdMan::launchCommand(AIGMan & aigman, const std::string & cmd, const std::vector<std::string> & vLiterals) {
    if (vLiterals.size() == 0 || vLiterals[0] != cmd) {
        printf("Wrong argument format of command %s!\n", cmd.c_str());
        return;
    }

    auto cmdRet = htCmdFuncs.find(cmd);
    if (cmdRet == htCmdFuncs.end()) {
        printf("Command %s not registered, ignored.\n", cmd.c_str());
        return;
    }

    auto & cmdHandler = cmdRet->second;
    int ret = cmdHandler(aigman, vLiterals);
    if (ret == 1) {
        printf("Command %s not recognized or in wrong format!\n", cmd.c_str());
        return;
    }
}

void CmdMan::cliMenuAddCommands(AIGMan & aigman, std::unique_ptr<cli::Menu> & cliMenu) {
    for (const auto & it : htCmdFuncs) {
        const auto & cmd = it.first;

        auto cliFunc = [&](std::ostream& out, std::vector<std::string> vTokens) {
            // vTokens provided by cli does not include cmd as the first element, add a new one
            std::vector<std::string> vLiterals;
            vLiterals.push_back(cmd);
            vLiterals.insert(vLiterals.end(), vTokens.begin(), vTokens.end());

            launchCommand(aigman, cmd, vLiterals);
        };

        cliMenu->Insert(cmd, cliFunc, cmd);
    }
}
