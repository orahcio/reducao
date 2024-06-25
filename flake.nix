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
      jupyter_bokeh = pkgs.python311Packages.buildPythonPackage rec {
        propagatedBuildInputs = with pkgs.python311Packages;[
          bokeh
          ipywidgets
        ];
        pname = "jupyter_bokeh";
        version = "4.0.5";
        format = "wheel";
        url = "https://files.pythonhosted.org/packages/99/c6/c4b923e6db17cfa52d3df37a0b7b0441e289d4e4ccfd0bc286ee51600ac5/jupyter_bokeh-3.0.7-py3-none-any.whl";
        src = pkgs.python311Packages.fetchPypi {
          inherit pname version;
          sha256 = "a33d6ab85588f13640b30765fa15d1111b055cbe44f67a65ca57d3593af8245d";
        };
      };
      # photutils = pkgs.python311Packages.buildPythonPackage rec {
      #   propagatedBuildInputs = with pkgs.python311Packages;[
      #     numpy
      #     astropy
      #     # Requires
      #     setuptools-scm
      #     cython
      #     astropy-extension-helpers
      #   ];
      #   pname = "photutils";
      #   version = "1.11.0";
      #   format = "pyproject";
      #   src = pkgs.python311Packages.fetchPypi {
      #     inherit pname version;
      #     sha256 = "ee709d090bac2bc8b8b317078c87f9c4b6855f6f94c2f7f2a93ab5b8f8375597";
      #   };
      # };
    in {
      devShell = pkgs.mkShell rec {
        venvDir = "./.venv";
        buildInputs = with pkgs.python311Packages; [
          venvShellHook
          jupyter_bokeh
          bokeh
          # colorcet
          flask
          numpy
          werkzeug
          astropy
          astroquery
          photutils
          statsmodels
          gunicorn
          pandas
          openpyxl
          # #poetry
          # pkgs.stdenv.cc.cc
          # pkgs.zlib
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
          export PYTHONPATH="$PIP_PREFIX/${pkgs.python311.sitePackages}:$PYTHONPATH"
          export PATH="$PIP_PREFIX/bin:$PATH"
          
          # for Numpy
          export LD_LIBRARY_PATH=${pkgs.zlib}/lib:$LD_LIBRARY_PATH

          # export LD_LIBRARY_PATH=${pkgs.glibc}/lib:$LD_LIBRARY_PATH

          export LD_LIBRARY_PATH=${pkgs.zlib}/lib:$LD_LIBRARY_PATH
          #   # GL libraries (for gym environment rendering)
          #   export LD_LIBRARY_PATH=${pkgs.libGL}/lib:$LD_LIBRARY_PATH
          #   export LD_LIBRARY_PATH=${pkgs.libGLU}/lib:$LD_LIBRARY_PATH

          unset SOURCE_DATE_EPOCH
          # source .venv/bin/activate
        '';
      };
    });
}
