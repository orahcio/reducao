{ pkgs ? import <nixpkgs> {} }:

with import <nixpkgs> {};
with pkgs.python3Packages;

let
  photutils = buildPythonPackage rec {
    propagatedBuildInputs = [ setuptools numpy astropy ];
    pname = "photutils";
    version = "1.6.0";
    format = "wheel";
    url = "https://files.pythonhosted.org/packages/1c/19/90df1a85f84fb9fa3725d7a1e0e23fc27fbd1d1939bf31f86b954b3a4c15/photutils-1.6.0-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl";
    #format = "wheel";
    src = fetchurl {
      inherit url;
      sha256 = "733ed51d42783f65e750ab77430c9cbfd5502420efd39927b6f6fde1023e5078";
      #dist = python;
      #python = "py3";
    };
  };

  my-python = pkgs.python3;
  python-with-my-packages = my-python.withPackages (p: with p; [
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
    photutils
  ]);
in
pkgs.mkShell {
  packages = [
    python-with-my-packages
  ];
}
