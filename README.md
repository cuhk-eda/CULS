# CULS
CULS is a GPU-based logic synthesis tool developed by the research team 
supervised by Prof. Evangeline F. Y. Young at The Chinese University of Hong Kong (CUHK).

## Dependencies
* CMake >= 3.8
* GCC >= 7.5.0
* CUDA >= 11.4

## Building
* Build as a standalone tool:
    ```bash
    mkdir build && cd build
    cmake ..
    make
    ```
    The built binary executable will be named `gpuls`. 

* Build as a patch of ABC:
    ```bash
    mkdir build && cd build
    cmake .. -DPATCH_ABC=1
    make
    ```
    The built binary executable will be named `abcg`. 

    If the readline library is installed in a custom path on your machine,
    add the option `-DREADLINE_ROOT_DIR=<readline_path>` when invoking cmake.
    CULS can still be successfully built even if the readline library 
    is not found.

## Getting started

* Standalone mode

    To interact with the command prompt, run
    ```bash
    ./gpuls
    ```
    
    You can also directly execute a script, e.g., 
    ```bash
    ./gpuls -c "read ../abc/i10.aig; resyn2; write i10_resyn2.aig"
    ```
* ABC patch mode

    The usage is the same as ABC. For instance, 
    ```bash
    ./abcg -c "read ../abc/i10.aig; gget; gresyn2; gput; print_stats; cec -n"
    ```

## Commands

* Standalone mode
    * `read`: read an AIG from a file
    * `write`: dump the internal AIG to a file
    * `b`: AIG balancing
    * `rw`: AIG rewriting
    * `rf`: AIG refactoring
    * `rs`: AIG resubstitution
    * `st`: strashing and dangling-node removal
    * `resyn2`: perform the resyn2 optimization script
    * `resyn2rs`: perform the resyn2rs optimization script
    * `ps`: print AIG statistics
    * `time`: print time statistics

* ABC patch mode

    Standalone mode commands will be prefixed by `g`, e.g., `grf`
    for AIG refactoring. 

    Additionally, there are two commands `gget` and `gput` for converting the
    AIG data structure from ABC to GPU, and from GPU to ABC, respectively,
    similar to the ABC9 package. 

## Publications
* Shiju Lin, Jinwei Liu, Tianji Liu, Martin D.F. Wong, Evangeline F.Y. Young, 
"NovelRewrite: Node-Level Parallel AIG Rewriting", 
59th ACM/IEEE Design Automation Conference (DAC), 2022.
* Tianji Liu, Evangeline F.Y. Young, "Rethinking AIG Resynthesis in Parallel", 
60th ACM/IEEE Design Automation Conference (DAC), 2023.
* Yang Sun, Tianji Liu, Martin D.F. Wong, Evangeline F.Y. Young, 
"Massively Parallel AIG Resubstitution", 
61st ACM/IEEE Design Automation Conference (DAC), 2024.
* Tianji Liu, Lei Chen, Xing Li, Mingxuan Yuan, Evangeline F.Y. Young, 
"FineMap: A Fine-grained GPU-parallel LUT Mapping Engine", 
29th Asia and South Pacific Design Automation Conference (ASP-DAC), 2024.
* Tianji Liu, Yang Sun, Lei Chen, Xing Li, Mingxuan Yuan, Evangeline F.Y. Young,
"A Unified Parallel Framework for LUT Mapping and Logic Optimization",
IEEE Transactions on Computer-Aided Design of Integrated Circuits 
and Systems (TCAD), 2024.
* Tianji Liu, Evangeline F.Y. Young, "Simulation-based Parallel Sweeping: 
A New Perspective on Combinational Equivalence Checking", 
62nd ACM/IEEE Design Automation Conference (DAC), 2025. 

## Other Algorithms Developed on Top of CULS
* GPU LUT mapping
* GPU mapping-based AIG optimization
* GPU simulation-based combinational equivalence checking

These algorithms are not open-sourced in CULS due to various reasons, but we can provide
binary executables containing their implementations. To request the executables, 
please send an email to Tianji Liu including your name, affiliation, 
and the intended use of the executable.

## Contributors
* [Shiju Lin](https://shijulin.github.io/): GPU rewriting.
* [Jinwei Liu](https://anticold.github.io/): GPU rewriting.
* [Tianji Liu](https://tefantasy.github.io/): GPU refactoring, balancing,
LUT mapping, mapping-based AIG optimization, GPU simulation-based CEC.
* Yang Sun: GPU resubstitution, mapping-based AIG optimization.
