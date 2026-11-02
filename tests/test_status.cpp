#include <catch2/catch_test_macros.hpp>
#include <string>
#include "cme/status.hpp"

TEST_CASE("Status::ok is ok with an empty message") {
    const cme::Status status = cme::Status::ok();
    CHECK(status.is_ok());
    CHECK(status.code() == cme::StatusCode::Ok);
    CHECK(status.message().empty());
}

TEST_CASE("Status::error rejects StatusCode::Ok") {
    const cme::Status status = cme::Status::error(cme::StatusCode::Ok, "ignored");
    CHECK_FALSE(status.is_ok());
    CHECK(status.code() == cme::StatusCode::InvalidArgument);
    CHECK(status.message() == "Status::error called with Ok");
}

TEST_CASE("Status::error retains code and message") {
    const cme::Status status =
        cme::Status::error(cme::StatusCode::DimensionMismatch, "A and B disagree");
    CHECK_FALSE(status.is_ok());
    CHECK(status.code() == cme::StatusCode::DimensionMismatch);
    CHECK(status.message() == "A and B disagree");
}
