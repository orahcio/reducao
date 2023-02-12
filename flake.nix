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
      packages.fhs-environment = pkgs.buildFHSUserEnv {
        name = "myEnv";
        targetPkgs = pkgs: with pkgs; [
          python3
          poetry
        ];
        runScript = "bash";
      };

      devShell = pkgs.mkShell {
        buildInputs = with pkgs; [
          self.packages.${system}.fhs-environment
        ];
        shellHook = ''
          # export PIP_PREFIX=$(pwd)/_build/pip_packages #Dir where built packages are stored
          # export PYTHONPATH="$PIP_PREFIX/${pkgs.python3.sitePackages}:$PYTHONPATH"
          # export PATH="$PIP_PREFIX/bin:$PATH"
          # unset SOURCE_DATE_EPOCH
          # source .venv/bin/activate
          exec myEnv
        '';
      };
    });
}
