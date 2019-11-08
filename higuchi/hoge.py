#Python2、Python3両方ともインストールし直す場合。------------------

# -upgradeでインストールされたpipをアンインストール。（No module named pipとでる場合はそのまま進む。）
sudo python -m pip uninstall pip
sudo python3 -m pip uninstall pip

# aptでインストールされたpipをアンインストール。
sudo apt autoremove python3-pip
sudo apt autoremove python-pip

# get-pip.pyをダウンロード。
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py

# pipを標準インストール。(get-pip.pyがhome dirにあるとして、home dirで実行)
python3 get-pip.py
python get-pip.py

# 「python3 get-pip.py」で「ModuleNotFoundError: No module named 'distutils.util'」と出てくる場合。
sudo apt install python3-distutils
# もう一度「python3 get-pip.py」を実行する。

#Python3のみインストールし直す場合。-------------------------------

radeでインストールされたpipをアンインストール。（No module named pipとでる場合はそのまま進む。）
sudo python3 -m pip uninstall pip

# aptでインストールされたpipをアンインストール。
sudo apt autoremove python3-pip

# get-pip.pyをダウンロード。
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py

# pipを標準インストール。(get-pip.pyがhome dirにあるとして、home dirで実行)
python3 get-pip.py

# 「python3 get-pip.py」で「ModuleNotFoundError: No module named 'distutils.util'」と出てくる場合。
sudo apt install python3-distutils
# もう一度「python3 get-pip.py」を実行する。