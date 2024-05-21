# push!(LOAD_PATH,"../src/")
using MPIKernelReco
using Documenter

DocMeta.setdocmeta!(MPIKernelReco, :DocTestSetup, :(using MPIKernelReco); recursive=true)

makedocs(sitename = "MPIKernelReco Documentation")
