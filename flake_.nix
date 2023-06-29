{
  description = "Redução de dados fotométricos";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, }: flake-utils.lib.eachDefaultSystem (system:
    let
      pkgs = nixpkgs.legacyPackages.${system};
      customOverrides = self: super: {
        # Overrides go here
        });
      };
      app = pkgs.poetry2nix.mkPoetryApplication {
        projectDir = ./.;
        overrides = [
          pkgs.poetry2nix.defaultPoetryOverrides
          customOverrides
        ];
      };

      packageName = "reducao";

    in {
      packages.${packageName} = app;

      defaultPackage = self.packages.${system}.${packageName};

      devShell = pkgs.mkShell {
        buildInputs = with pkgs; [ poetry ];
        inputsFrom = builtins.attrValues self.packages.${system};
        #shellHook = ''
          # export PIP_PREFIX=$(pwd)/_build/pip_packages #Dir where built packages are stored
          # export PYTHONPATH="$PIP_PREFIX/${pkgs.python3.sitePackages}:$PYTHONPATH"
          # export PATH="$PIP_PREFIX/bin:$PATH"
          # unset SOURCE_DATE_EPOCH
          # source .venv/bin/activate
        #  exec myEnv
        #'';
      };
    });
}
