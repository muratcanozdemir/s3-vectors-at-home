# Changelog

## [0.2.0](https://github.com/muratcanozdemir/s3-vectors-at-home/compare/v0.1.1...v0.2.0) (2026-11-28)


### Features

* consistent backups to a directory, and verified restores ([#26](https://github.com/muratcanozdemir/s3-vectors-at-home/issues/26)) ([979c515](https://github.com/muratcanozdemir/s3-vectors-at-home/commit/979c5159ad6f01a71083d5e2b09792a2fb5ca9c6))


### Performance Improvements

* approximate (HNSW) search and an inverted index for metadata filters ([#25](https://github.com/muratcanozdemir/s3-vectors-at-home/issues/25)) ([a0d5786](https://github.com/muratcanozdemir/s3-vectors-at-home/commit/a0d5786d435f60029cb9e82b4b39b7a1262888a2))
* split long documents into chunks and cap index memory per worker ([#23](https://github.com/muratcanozdemir/s3-vectors-at-home/issues/23)) ([f591208](https://github.com/muratcanozdemir/s3-vectors-at-home/commit/f591208b06800520c380eb2467d5fcd8d3f2b9fd))

## [0.1.1](https://github.com/muratcanozdemir/s3-vectors-at-home/compare/v0.1.0...v0.1.1) (2026-11-28)


### Bug Fixes

* **ci:** tag images 0.1.0, 0.1 and latest, without the v ([985f675](https://github.com/muratcanozdemir/s3-vectors-at-home/commit/985f675eb6cb92b444a8db0bea1ac678d4bfd0a5))
* **compose:** give MinIO the same credentials as the API ([985f675](https://github.com/muratcanozdemir/s3-vectors-at-home/commit/985f675eb6cb92b444a8db0bea1ac678d4bfd0a5))

## 0.1.0 (2026-11-28)


### Features

* Amazon S3 Vectors API, compatible with the AWS SDKs ([bf4e583](https://github.com/muratcanozdemir/s3-vectors-at-home/commit/bf4e5831de8d0dfab08eaaacab9bd9c2619eca90))
* batch bulk upload, cache FAISS index in memory, bound paging/top_k ([#5](https://github.com/muratcanozdemir/s3-vectors-at-home/issues/5)) ([3cdac5e](https://github.com/muratcanozdemir/s3-vectors-at-home/commit/3cdac5eb24a475a606d9fb991b45dc30c26a233f))
* container image and docker compose up for MinIO + API (and fix the wheel) ([01d617a](https://github.com/muratcanozdemir/s3-vectors-at-home/commit/01d617ad20357f5a3a3772a09a1ff0d0c67fed15))
* metadata, filtered search, and named indexes ([#9](https://github.com/muratcanozdemir/s3-vectors-at-home/issues/9)) ([c7e736b](https://github.com/muratcanozdemir/s3-vectors-at-home/commit/c7e736b31a413a5b5ce546911a5d93d507134843))
* register in Murat's software catalog, add LICENSE ([72bbbce](https://github.com/muratcanozdemir/s3-vectors-at-home/commit/72bbbceda4a54c9b49b06058789e9c078995507d))


### Bug Fixes

* 7 bugs found by new edge-case and property test suites ([68fd38c](https://github.com/muratcanozdemir/s3-vectors-at-home/commit/68fd38c7d393f9d66956f3cc4fe5b9f1d9226830))
* **ci:** replace removed minio/minio image with chainguard/minio ([66adaa9](https://github.com/muratcanozdemir/s3-vectors-at-home/commit/66adaa9595294cb268f64478bf7e1361c6a0599d))
* make index writes safe across processes with conditional PUTs ([#7](https://github.com/muratcanozdemir/s3-vectors-at-home/issues/7)) ([cea9670](https://github.com/muratcanozdemir/s3-vectors-at-home/commit/cea9670135065b5dcfd19f14f4ed82c19c8df201))
* repair broken deps/CI, fix search+index correctness bugs, add golden tests ([2fc03e6](https://github.com/muratcanozdemir/s3-vectors-at-home/commit/2fc03e6ddcea4fcf764b6cd72dee88f8091d6ba0))


### Performance Improvements

* embed with ONNX Runtime instead of PyTorch ([#11](https://github.com/muratcanozdemir/s3-vectors-at-home/issues/11)) ([8cb75dc](https://github.com/muratcanozdemir/s3-vectors-at-home/commit/8cb75dc9e9d9fd35ce35a89577452c8b778c929b))
* segmented index layout, so writes cost the same at any size ([#12](https://github.com/muratcanozdemir/s3-vectors-at-home/issues/12)) ([2ce6b59](https://github.com/muratcanozdemir/s3-vectors-at-home/commit/2ce6b59e1e29bb4b7db3b68d25b07152b480fc22))
