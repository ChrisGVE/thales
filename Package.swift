// swift-tools-version:5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "Thales",
    platforms: [
        .iOS(.v14),
        .macOS(.v11)
    ],
    products: [
        .library(
            name: "Thales",
            targets: ["Thales"]
        ),
    ],
    targets: [
        // C bridge target: exposes swift-bridge generated C types (RustStr, etc.)
        .target(
            name: "ThalesBridge",
            path: "swift/Sources/ThalesBridge",
            publicHeadersPath: "include",
            cSettings: [
                .headerSearchPath("include")
            ]
        ),
        // Swift wrapper target: depends on ThalesBridge for C type visibility
        .target(
            name: "Thales",
            dependencies: ["ThalesBridge"],
            path: "swift/Sources/Thales",
            exclude: ["include"],
            linkerSettings: [
                // Users must provide the path to libthales.a via LIBRARY_SEARCH_PATHS
                // in their Xcode project or via -L flag
                .linkedLibrary("thales"),
                .linkedLibrary("resolv"),
            ]
        ),
    ]
)
