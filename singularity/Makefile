
ifndef library
	library=dummydfsdfsgdf
endif

update:
	sudo singularity exec --writable env.dir/ pip3 install --force-reinstall --no-deps $(library)

edit:
	sudo singularity shell --writable env.dir/

build:
	sudo rm -rf env.dir && sudo singularity build --sandbox env.dir thecontainer.def

pkg:
	sudo singularity build --force env.sif env.dir/

send:
	scp env.sif gpuCMM:~/projects/patdbpredict/
