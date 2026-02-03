IMG=baseten/wan2.1:base
CONTAINER_NAME=wan2.1_dev
.PHONY: format docker_build clean

format:
	isort generate.py wan
	yapf -i -r *.py generate.py wan

download_model:
	hf download Wan-AI/Wan2.1-T2V-14B --local-dir ./Wan2.1-T2V-14B

docker_build:
	docker build -t $(IMG) -f docker/Dockerfile .

docker_run:
	docker rm -f $(CONTAINER_NAME) || true
	docker run -d \
		-v $(realpath ../):/workspace \
		-it --ipc=host --shm-size 32g \
		--entrypoint bash \
		--gpus all \
		--name $(CONTAINER_NAME) \
		$(IMG) \
		-c 'sleep infinity'

docker_push:
	docker push $(IMG)

install_dep:
	uv sync
	uv pip install dist/blite_tracing-0.1.0-py3-none-any.whl
# 	uv pip install dist/flash_attn-2.8.3+cu12torch2.9-cp312-cp312-linux_x86_64.whl

run_demo: install_dep
	ENABLE_PROFILE=1 BLITE_TRACING_ENABLED=1 uv run torchrun --nproc_per_node=4 generate.py \
	--task t2v-14B \
	--size 1280*720 \
	--ckpt_dir ./Wan2.1-T2V-14B \
	--dit_fsdp --t5_fsdp \
	--ulysses_size 4 \
	--prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
# 	ENABLE_B10_FP4_LINEAR=0 ENABLE_PROFILE=1 BLITE_TRACING_ENABLED=1 uv run torchrun --nproc_per_node=4 generate.py \
# 		--task t2v-A14B \
# 		--size '1280*720' \
# 		--ckpt_dir ./Wan2.2-T2V-A14B \
# 		--dit_fsdp --t5_fsdp \
# 		--ulysses_size 4 \
# 		--prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."

clean:
	rm -rf *.mp4 && rm -rf *.json && rm -rf *.json.gz

convert_fp4:
	python3 wan/utils/quantization/convert_weights_to_nvfp4.py \
		--model_path ./Wan2.2-T2V-A14B \
		--output_path ./Wan2.2-T2V-A14B-NVFP4 \
		--block_size 16 

run_fp4_demo:
	ENABLE_B10_FP4_LINEAR=1 ENABLE_PROFILE=1 BLITE_TRACING_ENABLED=1 torchrun --nproc_per_node=8 generate.py \
		--task t2v-A14B \
		--size '1280*720' \
		--ckpt_dir ./Wan2.2-T2V-A14B-NVFP4 \
		--ulysses_size 8 \
		--prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
