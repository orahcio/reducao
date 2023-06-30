{
  inputs = {
    nixpkgs = {
      url = "github:nixos/nixpkgs/nixos-unstable";
    };
    flake-utils = {
      url = "github:numtide/flake-utils";
    };
  };
  outputs = { self, nixpkgs, flake-utils, ... }: flake-utils.lib.eachDefaultSystem (system:
    let
      pkgs = import nixpkgs {
        inherit system;
      };
      my_astroquery = pkgs.python311Packages.buildPythonPackage rec {
        pname = "astroquery";
        version = "0.4.7.dev7761";
        src = pkgs.python311Packages.fetchPypi {
          inherit pname version;
          sha256 = "7f66e39f0d9e22c0b7bd355ae74218797d87eab9d7236f2fcb1537ee5da7ceda";
        };
        propagatedbuildInputs = with pkgs.python311Packages;[
          setuptools
        ];
        buildInputs = with pkgs.python311Packages;[
          astropy
          pyvo
          html5lib
          beautifulsoup4
          keyring
          pytest-astropy
          pytest
          pillow
          matplotlib
        ];
        doCheck = false;
      };
    in {
      devShell = pkgs.mkShell rec {
        venvDir = "./.venv";
        buildInputs = with pkgs.python311Packages; [
          venvShellHook
          bokeh
          colorcet
          flask
          numpy
          werkzeug
          astropy
          # keyring
          # beautifulsoup4
          (astroquery.overridePythonAttrs (_: { doCheck = false; }))
          statsmodels
          gunicorn
          pandas
          #poetry
          pkgs.stdenv.cc.cc
          pkgs.zlib
          #lib
          #libGL
          #libGLU
          #xorg.libX11
        ];
        # Esse approuch permite que eu instale pacotes que estão quebrados no nixos pelo pip
        shellHook = ''  
          # for PyTorch
          export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
          
          # Dir where built packages are stored
          export PIP_PREFIX=$(pwd)/_build/pip_packages
          export PYTHONPATH="$PIP_PREFIX/${pkgs.python3.sitePackages}:$PYTHONPATH"
          export PATH="$PIP_PREFIX/bin:$PATH"
          
          # for Numpy
          export LD_LIBRARY_PATH=${pkgs.zlib}/lib:$LD_LIBRARY_PATH

          # export LD_LIBRARY_PATH=${pkgs.glibc}/lib:$LD_LIBRARY_PATH

          export LD_LIBRARY_PATH=${pkgs.zlib}/lib:$LD_LIBRARY_PATH
          #   # GL libraries (for gym environment rendering)
          #   export LD_LIBRARY_PATH=${pkgs.libGL}/lib:$LD_LIBRARY_PATH
          #   export LD_LIBRARY_PATH=${pkgs.libGLU}/lib:$LD_LIBRARY_PATH

          unset SOURCE_DATE_EPOCH
          source .venv/bin/activate
          # pip install --user pipenv
          # pipenv install
        '';
      };
    });
}
