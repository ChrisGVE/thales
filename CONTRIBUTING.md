# Contributing to Thales

Thank you for your interest in contributing to Thales!

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/ChrisGVE/thales.git
   cd thales
   ```

2. Install Rust (if not already installed):
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

3. Build the project:
   ```bash
   cargo build
   ```

4. Run tests:
   ```bash
   cargo test
   ```

## Code Style

- Follow Rust's standard formatting guidelines
- Run `cargo fmt` before committing
- Run `cargo clippy` and address warnings
- Add documentation for public APIs

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with clear commit messages
3. Ensure all tests pass
4. Update documentation as needed
5. Submit a pull request

## Release Process

Releases are automated via GitHub Actions. To create a new release:

1. Update the version in `Cargo.toml`
2. Update `CHANGELOG.md` with the new version's changes
3. Commit the changes: `git commit -am "chore: bump version to X.Y.Z"`
4. Create and push a version tag:
   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

The release workflow will automatically:
- Verify the tag matches `Cargo.toml` version
- Run tests
- Create a GitHub release with changelog notes
- Publish to crates.io

### Required Secrets

For the release workflow to publish to crates.io, the repository must have the following secret configured:

- `CARGO_REGISTRY_TOKEN`: API token from crates.io with publish permissions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
