cd /github/workspace
apt install -y portaudio19-dev

wget https://github.com/sccn/liblsl/releases/download/v1.16.0/liblsl-1.16.0-jammy_amd64.deb
dpkg -i liblsl-1.16.0-jammy_amd64.deb
#mkdir /root/.pyenv/versions/3.9.15/lib/python3.9/site-packages/pylsl/lib
#cp /usr/lib/liblsl.so /root/.pyenv/versions/3.9.15/lib/python3.9/site-packages/pylsl/lib/liblsl.so
./entrypoint_linux.sh . https://pypi.python.org/ https://pypi.python.org/simple "timeflux_neurofeedback_inverse_gamepad.spec" "requirements_linux.txt" "requirements_uninstall_linux.txt"
