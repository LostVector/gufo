#include "src/core/platform/net.hpp"

#include <cerrno>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <system_error>

#if defined(_WIN32)
#include <mstcpip.h>
#endif

namespace gufo::net {
namespace {

#if defined(_WIN32)
void StartWinsock() {
  WSADATA data{};
  const int error = WSAStartup(MAKEWORD(2, 2), &data);
  if (error != 0)
    throw std::system_error(error, std::system_category(),
                            "initialize WinSock 2.2");
}
#endif

}  // namespace

void EnsureSocketRuntimeStarted() {
#if defined(_WIN32)
  static std::once_flag once;
  std::call_once(once, StartWinsock);
#endif
}

void CloseSocket(Socket socket) noexcept {
#if defined(_WIN32)
  closesocket(socket);
#else
  ::close(socket);
#endif
}

Socket CreateStreamSocket(int address_family) noexcept {
  return ::socket(address_family, SOCK_STREAM, 0);
}

int PollSocket(void* descriptors, std::uint32_t count, int timeout_ms) {
#if defined(_WIN32)
  return WSAPoll(static_cast<WSAPOLLFD*>(descriptors), count, timeout_ms);
#else
  return ::poll(static_cast<struct pollfd*>(descriptors), count, timeout_ms);
#endif
}

std::intptr_t SendNoSignal(Socket socket, const void* data, std::size_t bytes) {
#if defined(_WIN32)
  // Winsock's send takes an int length; clamp each call so oversized buffers
  // narrow safely. The send loops handle partial writes.
  const int chunk = static_cast<int>(std::min<std::size_t>(
      bytes, static_cast<std::size_t>(std::numeric_limits<int>::max())));
  return ::send(socket, static_cast<const char*>(data), chunk, 0);
#else
#ifdef MSG_NOSIGNAL
  return ::send(socket, data, bytes, MSG_NOSIGNAL);
#else
  return ::send(socket, data, bytes, 0);
#endif
#endif
}

bool SetNonBlocking(Socket socket) noexcept {
#if defined(_WIN32)
  u_long mode = 1;
  return ioctlsocket(socket, FIONBIO, &mode) == 0;
#else
  const int flags = ::fcntl(socket, F_GETFL, 0);
  if (flags < 0)
    return false;
  return ::fcntl(socket, F_SETFL, flags | O_NONBLOCK) == 0;
#endif
}

std::intptr_t Receive(Socket socket, void* data, std::size_t bytes,
                      int flags) noexcept {
#if defined(_WIN32)
  // Winsock's recv takes an int length; clamp each call so oversized reads
  // narrow safely.
  const int chunk = static_cast<int>(std::min<std::size_t>(
      bytes, static_cast<std::size_t>(std::numeric_limits<int>::max())));
  return ::recv(socket, static_cast<char*>(data), chunk, flags);
#else
  return ::recv(socket, data, bytes, flags);
#endif
}

bool ShutdownSocket(Socket socket, ShutdownHow how) noexcept {
#if defined(_WIN32)
  const int mode = how == ShutdownHow::kRead
                       ? SD_RECEIVE
                       : (how == ShutdownHow::kWrite ? SD_SEND : SD_BOTH);
  return ::shutdown(socket, mode) == 0;
#else
  const int mode = how == ShutdownHow::kRead
                       ? SHUT_RD
                       : (how == ShutdownHow::kWrite ? SHUT_WR : SHUT_RDWR);
  return ::shutdown(socket, mode) == 0;
#endif
}

Socket AcceptClientSocket(Socket listener) noexcept {
  const Socket client = ::accept(listener, nullptr, nullptr);
#if defined(_WIN32)
  if (client != kInvalidSocket && !SetBlocking(client)) {
    CloseSocket(client);
    return kInvalidSocket;
  }
#endif
  return client;
}

int PollReadable(Socket socket, int timeout_ms) noexcept {
#if defined(_WIN32)
  WSAPOLLFD pfd{.fd = socket, .events = POLLRDNORM, .revents = 0};
  return WSAPoll(&pfd, 1, timeout_ms);
#else
  struct pollfd pfd{.fd = socket, .events = POLLIN, .revents = 0};
  return ::poll(&pfd, 1, timeout_ms);
#endif
}

int PollWritable(Socket socket, int timeout_ms) noexcept {
#if defined(_WIN32)
  WSAPOLLFD pfd{.fd = socket, .events = POLLWRNORM, .revents = 0};
  return WSAPoll(&pfd, 1, timeout_ms);
#else
  struct pollfd pfd{.fd = socket, .events = POLLOUT, .revents = 0};
  return ::poll(&pfd, 1, timeout_ms);
#endif
}

bool IsSocketConnectPending() noexcept {
#if defined(_WIN32)
  const int error = WSAGetLastError();
  return error == WSAEWOULDBLOCK || error == WSAEINPROGRESS;
#else
  return errno == EINPROGRESS;
#endif
}

bool SocketHasNoError(Socket socket) noexcept {
  int error = 0;
  SocklenT size = sizeof(error);
#if defined(_WIN32)
  return ::getsockopt(socket, SOL_SOCKET, SO_ERROR,
                      reinterpret_cast<char*>(&error), &size) == 0 &&
         error == 0;
#else
  return ::getsockopt(socket, SOL_SOCKET, SO_ERROR, &error, &size) == 0 &&
         error == 0;
#endif
}

std::string AddressErrorMessage(int error) {
#if defined(_WIN32)
  return ::gai_strerrorA(error);
#else
  return ::gai_strerror(error);
#endif
}

bool SetKeepAlive(Socket socket, int idle_seconds, int interval_seconds,
                  int probes) noexcept {
  const int enabled = 1;
#if defined(_WIN32)
  // Enabling keepalive is portable; Winsock retains its system probe limits.
  (void)idle_seconds;
  (void)interval_seconds;
  (void)probes;
  return ::setsockopt(socket, SOL_SOCKET, SO_KEEPALIVE,
                      reinterpret_cast<const char*>(&enabled),
                      sizeof(enabled)) == 0;
#else
  const bool enabled_ok = ::setsockopt(socket, SOL_SOCKET, SO_KEEPALIVE,
                                       &enabled, sizeof(enabled)) == 0;
#if defined(TCP_KEEPIDLE) && defined(TCP_KEEPINTVL) && defined(TCP_KEEPCNT)
  // Preserve the control channel's one-minute idle, ten-second probes and
  // six-probe limit on Linux, including while a server is otherwise idle.
  (void)::setsockopt(socket, IPPROTO_TCP, TCP_KEEPIDLE, &idle_seconds,
                     sizeof(idle_seconds));
  (void)::setsockopt(socket, IPPROTO_TCP, TCP_KEEPINTVL, &interval_seconds,
                     sizeof(interval_seconds));
  (void)::setsockopt(socket, IPPROTO_TCP, TCP_KEEPCNT, &probes, sizeof(probes));
#else
  (void)idle_seconds;
  (void)interval_seconds;
  (void)probes;
#endif
  return enabled_ok;
#endif
}

bool SetReceiveTimeout(Socket socket, int timeout_ms) noexcept {
#if defined(_WIN32)
  const DWORD ms = static_cast<DWORD>(timeout_ms);
  return ::setsockopt(socket, SOL_SOCKET, SO_RCVTIMEO,
                      reinterpret_cast<const char*>(&ms), sizeof(ms)) == 0;
#else
  timeval timeout{};
  timeout.tv_sec = timeout_ms / 1000;
  timeout.tv_usec = (timeout_ms % 1000) * 1000;
  return ::setsockopt(socket, SOL_SOCKET, SO_RCVTIMEO, &timeout,
                      sizeof(timeout)) == 0;
#endif
}

PeerReceiveState QueryPeerReceiveState(Socket socket) noexcept {
  if (socket == kInvalidSocket)
    return PeerReceiveState::kClosed;
#if defined(_WIN32)
#if defined(SIO_TCP_INFO) && defined(NTDDI_WIN10_RS2) && \
    NTDDI_VERSION >= NTDDI_WIN10_RS2
  DWORD version = 0;
  DWORD returned = 0;
  TCP_INFO_v0 info{};
  if (WSAIoctl(socket, SIO_TCP_INFO, &version, sizeof(version), &info,
               sizeof(info), &returned, nullptr, nullptr) != 0)
    return PeerReceiveState::kError;
  if (returned < sizeof(info.State)) {
    WSASetLastError(WSAEINVAL);
    return PeerReceiveState::kError;
  }
  switch (info.State) {
    case TCPSTATE_CLOSED:
    case TCPSTATE_CLOSE_WAIT:
    case TCPSTATE_CLOSING:
    case TCPSTATE_LAST_ACK:
    case TCPSTATE_TIME_WAIT:
      return PeerReceiveState::kClosed;
    default:
      return PeerReceiveState::kOpen;
  }
#else
  WSASetLastError(WSAEOPNOTSUPP);
  return PeerReceiveState::kError;
#endif
#elif defined(POLLRDHUP)
  pollfd descriptor{.fd = socket, .events = POLLRDHUP, .revents = 0};
  if (::poll(&descriptor, 1, 0) < 0)
    return PeerReceiveState::kError;
  if ((descriptor.revents & POLLNVAL) != 0) {
    errno = EBADF;
    return PeerReceiveState::kError;
  }
  if ((descriptor.revents & POLLERR) != 0) {
    int error = 0;
    socklen_t size = sizeof(error);
    if (::getsockopt(socket, SOL_SOCKET, SO_ERROR, &error, &size) == 0)
      errno = error != 0 ? error : ECONNRESET;
    return PeerReceiveState::kError;
  }
  return (descriptor.revents & (POLLRDHUP | POLLHUP)) != 0
             ? PeerReceiveState::kClosed
             : PeerReceiveState::kOpen;
#else
  errno = ENOTSUP;
  return PeerReceiveState::kError;
#endif
}

bool IsPeerDisconnected(Socket socket) noexcept {
  if (socket == kInvalidSocket)
    return true;
#if defined(_WIN32)
  WSAPOLLFD pfd{.fd = socket, .events = POLLRDNORM, .revents = 0};
  const int polled = WSAPoll(&pfd, 1, 0);
  if (polled < 0)
    return true;
  if (polled == 0)
    return false;
  if ((pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0)
    return true;
  if ((pfd.revents & POLLIN) == 0)
    return false;
  u_long available = 0;
  if (ioctlsocket(socket, FIONREAD, &available) == 0 && available > 0)
    return false;
  char byte = 0;
  const int n = ::recv(socket, &byte, 1, MSG_PEEK);
  if (n == 0)
    return true;
  if (n < 0) {
    const int err = WSAGetLastError();
    return err != WSAEWOULDBLOCK && err != WSAEINTR && err != WSAETIMEDOUT;
  }
  return false;
#else
  struct pollfd pfd{.fd = socket, .events = POLLIN, .revents = 0};
  if (::poll(&pfd, 1, 0) <= 0)
    return false;
  if ((pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0)
    return true;
  if ((pfd.revents & POLLIN) == 0)
    return false;
  char byte = 0;
  const ssize_t n = ::recv(socket, &byte, 1, MSG_PEEK | MSG_DONTWAIT);
  return n == 0 ||
         (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR);
#endif
}

int LastSocketError() noexcept {
#if defined(_WIN32)
  return WSAGetLastError();
#else
  return errno;
#endif
}

bool IsSocketInterrupted() noexcept {
#if defined(_WIN32)
  return WSAGetLastError() == WSAEINTR;
#else
  return errno == EINTR;
#endif
}

bool IsSocketWouldBlockOrTimeout() noexcept {
#if defined(_WIN32)
  const int err = WSAGetLastError();
  return err == WSAEWOULDBLOCK || err == WSAETIMEDOUT;
#else
  return errno == EAGAIN || errno == EWOULDBLOCK;
#endif
}

bool IsSocketAcceptRetryable() noexcept {
#if defined(_WIN32)
  const int err = WSAGetLastError();
  return err == WSAEINTR || err == WSAEWOULDBLOCK || err == WSAECONNRESET;
#else
  return errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK ||
         errno == ECONNABORTED;
#endif
}

bool SetReuseAddr(Socket socket) noexcept {
  int opt = 1;
  return ::setsockopt(socket, SOL_SOCKET, SO_REUSEADDR,
                      reinterpret_cast<const char*>(&opt), sizeof(opt)) == 0;
}

bool SetExclusiveListenerReuse(Socket socket) noexcept {
  int opt = 1;
#if defined(_WIN32)
  return ::setsockopt(socket, SOL_SOCKET, SO_EXCLUSIVEADDRUSE,
                      reinterpret_cast<const char*>(&opt), sizeof(opt)) == 0;
#else
  (void)::setsockopt(socket, SOL_SOCKET, SO_REUSEADDR,
                     reinterpret_cast<const char*>(&opt), sizeof(opt));
  return true;
#endif
}

bool ClearIpv6Only(Socket socket) noexcept {
  int opt = 0;
  return ::setsockopt(socket, IPPROTO_IPV6, IPV6_V6ONLY,
                      reinterpret_cast<const char*>(&opt), sizeof(opt)) == 0;
}

bool SetBlocking(Socket socket) noexcept {
#if defined(_WIN32)
  u_long mode = 0;
  return ioctlsocket(socket, FIONBIO, &mode) == 0;
#else
  const int flags = ::fcntl(socket, F_GETFL, 0);
  if (flags < 0)
    return false;
  return ::fcntl(socket, F_SETFL, flags & ~O_NONBLOCK) == 0;
#endif
}

bool SetNoDelay(Socket socket) noexcept {
  int value = 1;
  return ::setsockopt(socket, IPPROTO_TCP, TCP_NODELAY,
                      reinterpret_cast<const char*>(&value),
                      sizeof(value)) == 0;
}

bool SetSocketTimeouts(Socket socket, int timeout_ms) noexcept {
#if defined(_WIN32)
  const DWORD ms = static_cast<DWORD>(timeout_ms);
  const bool received =
      setsockopt(socket, SOL_SOCKET, SO_RCVTIMEO,
                 reinterpret_cast<const char*>(&ms), sizeof(ms)) == 0;
  const bool sent =
      setsockopt(socket, SOL_SOCKET, SO_SNDTIMEO,
                 reinterpret_cast<const char*>(&ms), sizeof(ms)) == 0;
  return received && sent;
#else
  timeval tv{};
  tv.tv_sec = timeout_ms / 1000;
  tv.tv_usec = (timeout_ms % 1000) * 1000;
  const bool received =
      ::setsockopt(socket, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) == 0;
  const bool sent =
      ::setsockopt(socket, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv)) == 0;
  return received && sent;
#endif
}

bool SetSendTimeout(Socket socket, int timeout_ms) noexcept {
#if defined(_WIN32)
  DWORD ms = static_cast<DWORD>(timeout_ms);
  return setsockopt(socket, SOL_SOCKET, SO_SNDTIMEO,
                    reinterpret_cast<const char*>(&ms), sizeof(ms)) == 0;
#else
  timeval tv{};
  tv.tv_sec = timeout_ms / 1000;
  tv.tv_usec = (timeout_ms % 1000) * 1000;
  return ::setsockopt(socket, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv)) == 0;
#endif
}

long long SocketRecvTimeoutMs(Socket socket) noexcept {
#if defined(_WIN32)
  DWORD ms = 0;
  int size = sizeof(ms);
  if (getsockopt(socket, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<char*>(&ms),
                 &size) != 0)
    return -1;
  return static_cast<long long>(ms);
#else
  timeval tv{};
  socklen_t size = sizeof(tv);
  if (::getsockopt(socket, SOL_SOCKET, SO_RCVTIMEO, &tv, &size) != 0)
    return -1;
  return static_cast<long long>(tv.tv_sec) * 1000 + tv.tv_usec / 1000;
#endif
}

}  // namespace gufo::net
