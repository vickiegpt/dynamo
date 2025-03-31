use crate::tokens::Token;

pub enum BlockState {
    Unregistered,
    Partial(PartialState),
    Complete(CompleteState),
    Registered(RegisteredState),
}

impl BlockState {
    pub fn push_token(&mut self, token: Token) -> Result<(), Token> {
        match self {
            BlockState::Unregistered => Err(token),
            BlockState::Partial(partial) => match partial.push_token(token) {
                None => Ok(()),
                Some(block) => {
                    *self = BlockState::Complete(CompleteState {
                        sequence: partial.sequence,
                    });
                    Ok(())
                }
            },
            BlockState::Complete(_complete) => Err(token),
            BlockState::Registered(_registered) => Err(token),
        }
    }

    pub fn register(&mut self) -> Result<(), BlockError> {
        match self {
            BlockState::Unregistered | BlockState::Partial(_) => Err(BlockError::Unregistered),
            BlockState::Complete(complete) => Ok(BlockState::Registered(RegisteredState {
                sequence_hash: complete.sequence.sequence_hash(),
                block_hash: complete.sequence.block_hash(),
            })),
        }
    }
}

pub struct PartialState {
    sequence: TokenSequence,
}

impl PartialState {
    pub fn push_token(&mut self, token: Token) -> Result<(), BlockError> {}
}

pub struct CompleteState {
    sequence: TokenSequence,
}

impl CompleteState {}

pub struct RegisteredState {
    sequence_hash: u64,
    block_hash: u64,
}

impl RegisteredState {}
