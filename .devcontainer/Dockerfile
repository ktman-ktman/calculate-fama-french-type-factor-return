ARG VARIANT="latest"

FROM python:${VARIANT}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    curl \
    fish \
    git \
    git-lfs \
    htop \
    sudo \
    tmux \
    wget \
    && apt-get -y clean \
    && rm -rf /var/lib/apt/lists/*

# install poetry
ENV POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 

RUN curl -sSL https://install.python-poetry.org | python3 -

# user
ARG USERNAME="vscode"
ARG USER_UID="1000"
ARG USER_GID=${USER_UID}

RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} \
    && echo ${USERNAME} ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}

# make directory for extension
RUN mkdir -p /home/${USERNAME}/.vscode-server/extensions \
        /home/${USERNAME}/.vscode-server-insiders/extensions \
    && chown -R ${USERNAME} \
        /home/${USERNAME}/.vscode-server \
        /home/${USERNAME}/.vscode-server-insiders

CMD ["/bin/bash"]