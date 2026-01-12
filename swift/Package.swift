// swift-tools-version:5.7
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
        .target(
            name: "Thales",
            path: "Sources/Thales",
            publicHeadersPath: "include",
            cSettings: [
                .headerSearchPath("include")
            ],
            linkerSettings: [
                // Users must provide the path to libthales.a via LIBRARY_SEARCH_PATHS
                // in their Xcode project or via -L flag
                .linkedLibrary("thales"),
                .linkedLibrary("resolv"),  // Required by swift-bridge
            ]
        ),
    ]
)
