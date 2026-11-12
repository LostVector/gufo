{
  lib,
  newScope,
  pkgs,
  releaseVersion ? null,
  revision,
  rocmPackages,
  rustToolchain,
}:

lib.makeScope newScope (self: {
  gufo = self.callPackage ./package.nix { inherit releaseVersion revision rocmPackages rustToolchain; };
  tp2-rdma = self.callPackage ./package.nix {
    inherit releaseVersion revision rocmPackages rustToolchain;
    enableTp2Rdma = true;
    rdma-core = pkgs.rdma-core;
  };
  mkServe = self.callPackage ./mk-serve.nix { };
  extensionModules = self.callPackage ./extension-modules.nix { };
})
