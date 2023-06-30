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
      my_photutils = pkgs.python311Packages.buildPythonPackage rec {
        propagatedBuildInputs = with pkgs.python311Packages;[
          setuptools
          numpy
          astropy
        ];
        pname = "photutils";
        version = "1.8.0";
        format = "wheel";
        url = "https://files.pythonhosted.org/packages/42/15/c9a6126eda18d7e3f75da6f8302619f2d83dadb2f38c18e1fb221f606bf9/photutils-1.8.0-cp311-cp311-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl";
        src = pkgs.fetchurl { # pkgs.python311Packages.fetchPypi {
          inherit url; # pname version;
          sha256 = "6ce5ff0c03c4c426671da4a6a738279682f1dc2a352c77006561debc403955b7";
        };
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
          (astroquery.overridePythonAttrs (_: { doCheck = false; }))
          my_photutils
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
        '';
      };
    });
}
