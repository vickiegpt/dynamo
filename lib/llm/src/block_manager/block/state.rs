use crate::{block_manager::events::RegistrationHandle, tokens::{TokenBlock, TokenSequence}};

#[derive(Debug)]
pub enum BlockState {
    Reset,
    Partial(PartialState),
    Complete(CompleteState),
    Registered(RegisteredState),
}

impl BlockState {
    // pub fn push_token(&mut self, token: Token) -> Result<(), Token> {
    //     match self {
    //         BlockState::Unregistered => Err(token),
    //         BlockState::Partial(partial) => match partial.sequence.push_token(token) {
    //             None => Ok(()),
    //             Some(block) => {
    //                 *self = BlockState::Complete(CompleteState {
    //                     sequence: partial.sequence,
    //                 });
    //                 Ok(())
    //             }
    //         },
    //         BlockState::Complete(_complete) => Err(token),
    //         BlockState::Registered(_registered) => Err(token),
    //     }
    // }

    // pub fn register(&mut self) -> Result<(), BlockError> {
    //     match self {
    //         BlockState::Unregistered | BlockState::Partial(_) => Err(BlockError::Unregistered),
    //         BlockState::Complete(complete) => Ok(BlockState::Registered(RegisteredState {
    //             sequence_hash: complete.sequence.sequence_hash(),
    //             block_hash: complete.sequence.block_hash(),
    //         })),
    //     }
    // }
}

#[derive(Debug)]
pub struct PartialState {
    pub sequence: TokenSequence,
    pub salt: u64,
}

#[derive(Debug)]
pub struct CompleteState {
    pub token_block: TokenBlock,
    pub salt: u64,
}

#[derive(Debug)]
pub struct RegisteredState {
    pub sequence_hash: u64,
    pub salt: u64,
    registration_handle: Arc<RegistrationHandle>,
}
