# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

FROM nvcr.io/nvidia/pytorch:20.12-py3

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN pip install imageio-ffmpeg==0.4.3 pyspng==0.1.0
RUN pip install imutils wandb lpips
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
#RUN pip uninstall opencv-python -y
#RUN pip uninstall opencv-contrib-python -y
#RUN pip uninstall opencv-contrib-python-headless -y
RUN pip install opencv-python==4.5.5.64
RUN python -m pip install paddlepaddle==2.2.2 -i https://mirror.baidu.com/pypi/simple

# RUN pip install paddlepaddle-gpu==2.2.2.post110 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
RUN pip install paddleseg==2.4.0
RUN apt-get install ffmpeg libsm6 libxext6  -y


WORKDIR /workspace

# Unset TORCH_CUDA_ARCH_LIST and exec.  This makes pytorch run-time
# extension builds significantly faster as we only compile for the
# currently active GPU configuration.
RUN (printf '#!/bin/bash\nunset TORCH_CUDA_ARCH_LIST\nexec \"$@\"\n' >> /entry.sh) && chmod a+x /entry.sh
ENTRYPOINT ["/entry.sh"]
