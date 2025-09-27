# PR #3235 ActiveMessage System - Comprehensive Mitigation Plan

## Review Summary

### 1. CI/Build Issues
- **Doc tests failing** (exit code 101 in pre-merge-rust)
- **GitLab pipeline failure**
- **PR template missing required sections**
- **Docstring coverage critically low** (4.40% vs 80% required)

### 2. API Ergonomics Issues (ryanolson's feedback)
Critical API design issues that need addressing:

#### a) ACK System Redesign
- **Current**: Manual ACK registration with UUID tracking, JSON payloads
- **Requested**:
  - ACKs integrated into handler interface
  - `AckSender` object passed to handlers when ACK requested
  - Support for ACK, SingleResponse, StreamResponse patterns
  - Metadata should be private to implementation

#### b) Message Building & Payload
- **Current**: Raw `Bytes` payload, manual JSON construction
- **Requested**:
  - Builder pattern for message creation
  - Accept `T: Serialize` instead of `Bytes`
  - Strong types with builders for internal messages (not JSON)

#### c) Handler Interface
- **Current**: `Arc<dyn ActiveMessageClient>` passed to handlers
- **Requested**: Just `ActiveMessageClient` (no Arc/dyn)

#### d) Addressing
- **Current**: UUID-based `InstanceId` for sender
- **Requested**: `SenderAddress` with IP/Port information

#### e) Handler Naming
- **Current**: Simple string names like "compute"
- **Requested**: Hierarchical names like "module.submod.fnname"

#### f) Transport Detection
- **Current**: Automatic IPC detection for same-host
- **Issue**: Forked processes share IP but need TCP
- **Solution**: Default to TCP, only use IPC when certain

## Phased Implementation Plan

### Phase 1: Critical Fixes (Immediate - 2 hours)
Fix CI/build issues to unblock PR

#### Session 1.1: Fix Doc Tests (30 mins)
1. Investigate doc test failure in `active_message.md`
2. Fix code examples in documentation
3. Run `cargo test --doc` locally
4. Push fix

#### Session 1.2: Add Essential Documentation (1.5 hours)
Add minimal doc comments to reach ~40% coverage:
- Core traits in `handler.rs`, `client.rs`, `manager.rs`
- Main types in `zmq/manager.rs`, `zmq/client.rs`
- Public functions in `discovery.rs`

### Phase 2: ACK System Redesign (4 hours)

#### Session 2.1: Design New ACK API (1 hour)
1. Design `AckSender` trait and implementation
2. Design response type enum (Ack, SingleResponse, StreamResponse)
3. Create internal message types with builders

#### Session 2.2: Implement ACK Handler Integration (3 hours)
```rust
// New handler signature
pub trait ActiveMessageHandler {
    async fn handle(
        &self,
        message: ActiveMessage,
        client: &ActiveMessageClient,
        responder: Option<Responder>,
    ) -> Result<()>;
}

pub enum Responder {
    Ack(AckSender),
    Single(SingleResponseSender),
    Stream(StreamResponseSender),
}
```

### Phase 3: Message Builder Pattern (3 hours)

#### Session 3.1: Message Builder Implementation (2 hours)
```rust
pub struct ActiveMessageBuilder<T: Serialize> {
    handler_name: String,
    payload: T,
    response_type: ResponseType,
}

impl<T: Serialize> ActiveMessageBuilder<T> {
    pub fn new(handler: impl Into<String>, payload: T) -> Self;
    pub fn expect_ack(self) -> Self;
    pub fn expect_response<R: DeserializeOwned>(self) -> Self;
    pub fn build(self) -> ActiveMessage;
}
```

#### Session 3.2: Internal Message Types (1 hour)
Create strongly-typed messages for system handlers:
```rust
pub struct RegisterServiceRequest {
    pub instance_id: Uuid,
    pub endpoint: Endpoint,
}

pub struct WaitForHandlerRequest {
    pub handler_name: String,
    pub timeout_ms: Option<u64>,
}
// etc...
```

### Phase 4: Address & Naming Improvements (2 hours)

#### Session 4.1: SenderAddress Implementation (1 hour)
```rust
pub struct SenderAddress {
    pub instance_id: Uuid,
    pub ip: IpAddr,
    pub port: u16,
}
```

#### Session 4.2: Hierarchical Handler Names (1 hour)
- Implement validation for dotted names
- Update examples to use hierarchical names
- Add name parsing utilities

### Phase 5: Transport & Examples (2 hours)

#### Session 5.1: Fix Transport Detection (1 hour)
- Default to TCP
- Add explicit IPC opt-in mechanism
- Document forked process considerations

#### Session 5.2: Update Examples (1 hour)
- Leader example with multi-worker ACK await
- Show new builder patterns
- Demonstrate response types

### Phase 6: Documentation & PR Update (2 hours)

#### Session 6.1: Complete Documentation (1.5 hours)
- Add comprehensive doc comments to all public APIs
- Include examples in doc comments
- Document migration from old to new API

#### Session 6.2: PR Finalization (30 mins)
- Update PR description with template sections
- Run all checks (clippy, fmt, tests, doc tests)
- Verify 80% documentation coverage
- Update migration guide

## Total Estimated Time: ~16 hours

## Priority Order

### Must Have (for PR approval):
1. Fix CI/doc tests
2. Basic documentation (40% minimum)
3. PR template update
4. ACK system redesign (core functionality)

### Should Have (significant improvements):
5. Message builder pattern
6. Strong typing for internal messages
7. Transport detection fix

### Nice to Have (can be follow-up PR):
8. SenderAddress refactor
9. Hierarchical handler names
10. Complete 80% documentation

## Breaking Changes

This refactor will introduce breaking changes:
- Handler trait signature change
- Message construction API change
- Client interface simplification
- ACK registration pattern change

## Migration Path

Provide migration guide showing:
- Old API vs New API examples
- Step-by-step migration instructions
- Compatibility layer (if feasible)

## Success Criteria

1. All CI checks passing
2. Documentation coverage ≥ 40% (minimum), ideally 80%
3. PR template properly filled
4. Core ACK ergonomics addressed
5. Examples demonstrate new patterns
6. Migration guide included