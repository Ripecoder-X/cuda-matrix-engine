#include "cme/status.hpp"

#include <utility>

namespace cme {

Status::Status(StatusCode code, std::string message)
    : code_(code), message_(std::move(message)) {}

Status Status::ok() { return Status(StatusCode::Ok, {}); }

Status Status::error(StatusCode code, std::string message) {
    if (code == StatusCode::Ok) {
        return Status(StatusCode::InvalidArgument, "Status::error called with Ok");
    }
    return Status(code, std::move(message));
}

bool Status::is_ok() const noexcept { return code_ == StatusCode::Ok; }

StatusCode Status::code() const noexcept { return code_; }

const std::string& Status::message() const noexcept { return message_; }

}  // namespace cme
