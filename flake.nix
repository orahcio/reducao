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
    in {
      devShell = pkgs.mkShell rec {
        venvDir = "./.venv";
        buildInputs = with pkgs.python3Packages; [
          venvShellHook
          bokeh
          colorcet
          flask
          numpy
          werkzeug
          astropy
          astroquery
          statsmodels
          pandas
          gunicorn
          #poetry
          pkgs.stdenv.cc.cc
          pkgs.zlib
          #lib
          #libGL
          #libGLU
          #xorg.libX11
        ];
        shellHook = ''
          # for PyTorch
          export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib

          # for Numpy
          export LD_LIBRARY_PATH=${pkgs.zlib}/lib:$LD_LIBRARY_PATH

        #   # GL libraries (for gym environment rendering)
        #   export LD_LIBRARY_PATH=${pkgs.libGL}/lib:$LD_LIBRARY_PATH
        #   export LD_LIBRARY_PATH=${pkgs.libGLU}/lib:$LD_LIBRARY_PATH
        '';
      };
    });
}
