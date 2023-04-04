#pragma once
#include <cryptopp/sha.h>
#include <cryptopp/hex.h>
#include "Helper.h"

[[nodiscard]] std::string getChecksum(const std::filesystem::path& path)
{
	assertm(std::filesystem::exists(path), path.string() + "doesn't exist.");
    auto file = std::ifstream{path, std::ios::binary};
    auto buffer = std::vector<CryptoPP::byte>(std::filesystem::file_size(path));
    auto sha256 = CryptoPP::SHA256{};
	// Read and update state
	file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
	sha256.Update(buffer.data(), file.gcount());
	// Create digest
    auto digest = std::string{};
	auto digestBytes = std::vector<CryptoPP::byte>(sha256.DigestSize());
    sha256.Final(digestBytes.data());
	// To hex string
	auto encoder = CryptoPP::HexEncoder{};
	auto sink = CryptoPP::StringSink{digest};
	encoder.Attach(new CryptoPP::Redirector{sink});
	encoder.Put(digestBytes.data(), sha256.DigestSize());
	encoder.MessageEnd();
	return digest;
}

