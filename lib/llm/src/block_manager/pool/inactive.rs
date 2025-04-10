use super::*;

impl<T: Storage, M: BlockMetadata> InactiveBlockPool<T, M> {
    pub async fn new() -> Self {
        let (match_tx, match_rx) = mpsc::unbounded_channel();
        let (return_tx, return_rx) = mpsc::unbounded_channel();
        let (control_tx, control_rx) = mpsc::unbounded_channel();
        let (fence_tx, fence_rx) = mpsc::unbounded_channel();

        let return_tx_clone = return_tx.clone();
        let return_handle = Arc::new(ReturnHandleImpl {
            return_tx: return_tx_clone,
        });

        let state = BlockPoolInner::new();

        let total_blocks_rx = state.total_blocks_watcher();
        let available_blocks_rx = state.available_blocks_watcher();

        let join_handle = tokio::spawn(progress_engine(
            match_rx, return_rx, control_rx, fence_rx, state,
        ));

        Self {
            match_tx,
            control_tx,
            fence_tx,
            return_handle,
            total_blocks_rx,
            available_blocks_rx,
            join_handle,
        }
    }

    pub fn total_blocks(&self) -> u64 {
        *self.total_blocks_rx.borrow()
    }

    pub fn available_blocks(&self) -> u64 {
        *self.available_blocks_rx.borrow()
    }

    pub fn total_blocks_watch(&self) -> watch::Receiver<u64> {
        self.total_blocks_rx.clone()
    }

    pub fn available_blocks_watch(&self) -> watch::Receiver<u64> {
        self.available_blocks_rx.clone()
    }

    pub fn is_active(&self) -> bool {
        !self.join_handle.is_finished()
    }

    pub async fn match_sequence_hashes(
        &self,
        hashes: Vec<SequenceHash>,
    ) -> Result<Vec<PoolItem<Block<T, M>>>> {
        let (tx, rx) = oneshot::channel();
        if self
            .match_tx
            .send(MatchRequest::MatchMultiple(MatchMultiple {
                hashes,
                return_handle: self.return_handle.clone(),
                tx,
            }))
            .is_err()
        {
            raise!("failed to send match request; channel closed");
        }

        let matched_blocks = rx.await?;
        Ok(matched_blocks)
    }

    pub async fn match_token_blocks(
        &self,
        token_blocks: &[TokenBlock],
    ) -> Result<Vec<PoolItem<Block<T, M>>>> {
        let hashes: Vec<u64> = token_blocks.iter().map(|b| b.sequence_hash()).collect();
        self.match_sequence_hashes(hashes).await
    }

    pub async fn take_blocks(&self, count: u32) -> Result<Vec<PoolItem<Block<T, M>>>> {
        let (tx, rx) = oneshot::channel();
        if self
            .match_tx
            .send(MatchRequest::Take(Take {
                count,
                return_handle: self.return_handle.clone(),
                tx,
            }))
            .is_err()
        {
            raise!("failed to send take request; channel closed");
        }

        let matched_blocks = rx.await?;
        Ok(matched_blocks)
    }

    pub async fn insert(&self, block: Block<T, M>) -> Result<()> {
        let (tx, rx) = oneshot::channel();
        if self
            .control_tx
            .send(ControlRequest::Insert(InsertControl { block, tx }))
            .is_err()
        {
            raise!("failed to send insert request; channel closed");
        }
        rx.await?;
        Ok(())
    }

    pub async fn update_single(&self, update: UpdateBlock<M>) -> Result<()> {
        let (tx, rx) = oneshot::channel();
        if self
            .control_tx
            .send(ControlRequest::UpdateSingle(UpdateSingleControl {
                update,
                tx,
            }))
            .is_err()
        {
            raise!("failed to send update single request; channel closed");
        }
        rx.await?;
        Ok(())
    }

    pub async fn update_multiple(&self, updates: Vec<UpdateBlock<M>>) -> Result<()> {
        let (tx, rx) = oneshot::channel();
        if self
            .control_tx
            .send(ControlRequest::UpdateMultiple(UpdateMultipleControl {
                updates,
                tx,
            }))
            .is_err()
        {
            raise!("failed to send update multiple request; channel closed");
        }
        rx.await?;
        Ok(())
    }

    pub async fn reset(&self, sequence_hashes: Vec<SequenceHash>) -> Result<()> {
        let (tx, rx) = oneshot::channel();
        if self
            .control_tx
            .send(ControlRequest::Reset(ResetControl {
                sequence_hashes,
                tx,
                _phantom: std::marker::PhantomData,
            }))
            .is_err()
        {
            raise!("failed to send reset request; channel closed");
        }
        rx.await?;
        Ok(())
    }

    pub async fn reset_all(&self) -> Result<()> {
        let (tx, rx) = oneshot::channel();
        if self
            .control_tx
            .send(ControlRequest::ResetAll(ResetAllControl {
                tx,
                _phantom: std::marker::PhantomData,
            }))
            .is_err()
        {
            raise!("failed to send reset all request; channel closed");
        }
        rx.await?;
        Ok(())
    }

    pub async fn fence(&self) -> Result<()> {
        let (tx, rx) = oneshot::channel();
        if self.fence_tx.send(tx).is_err() {
            raise!("failed to send fence request; channel closed");
        }
        rx.await?;
        Ok(())
    }
}

impl<T: Storage, M: BlockMetadata> BlockPoolInner<T, M> {
    fn new() -> Self {
        let (total_blocks_tx, _) = watch::channel(0);
        let (available_blocks_tx, _) = watch::channel(0);

        Self {
            lookup_map: HashMap::new(),
            priority_set: BTreeSet::new(),
            uninitialized_set: VecDeque::new(),
            return_tick: 0,
            total_blocks_tx,
            available_blocks_tx,
        }
    }

    pub fn total_blocks_watcher(&self) -> watch::Receiver<u64> {
        self.total_blocks_tx.subscribe()
    }

    pub fn available_blocks_watcher(&self) -> watch::Receiver<u64> {
        self.available_blocks_tx.subscribe()
    }

    fn insert_with_sequence_hash(
        &mut self,
        block: PoolValue<Block<T, M>>,
        sequence_hash: SequenceHash,
    ) {
        let priority_key = PriorityKey::new(block.metadata().clone(), sequence_hash);
        if self.priority_set.contains(&priority_key) {
            tracing::debug!("multiple entries with the same priority key, resetting block and inserting into uninitialized set");
            let mut block = block;
            block.reset();
            self.uninitialized_set.push_back(block);
        } else if let std::collections::hash_map::Entry::Vacant(e) =
            self.lookup_map.entry(sequence_hash)
        {
            tracing::debug!("inserting block to map and priority set");
            self.priority_set.insert(priority_key);
            e.insert(block);
        } else {
            tracing::debug!("multiple entries in lookup map with the same sequence hash, inserting into uninitialized set");
            let mut block = block;
            block.reset();
            self.uninitialized_set.push_back(block);
        }
    }

    // Insert an item with a given key and sequence_hash
    fn insert(&mut self, block: PoolValue<Block<T, M>>) {
        tracing::debug!("inserting block into available blocks");

        // If we already have an entry for this sequence hash or the block is reset,
        // we need to move it to the uninitialized set
        match block.state() {
            BlockState::Reset => {
                tracing::debug!("inserted block to uninitialized set");
                self.uninitialized_set.push_back(block);
            }
            BlockState::Partial(_) => {
                tracing::debug!("inserted block to uninitialized set");
                let mut block = block;
                block.reset();
                self.uninitialized_set.push_back(block);
            }
            BlockState::Complete(state) => {
                tracing::debug!("inserting completed/unregistered block to map and priority set");
                let sequence_hash = state.token_block.sequence_hash();
                self.insert_with_sequence_hash(block, sequence_hash);
            }
            BlockState::Registered(state) => {
                tracing::debug!("inserting registered block to map and priority set");
                let sequence_hash = state.sequence_hash;
                self.insert_with_sequence_hash(block, sequence_hash);
            }
        }
    }

    fn take_with_sequence_hash(
        &mut self,
        sequence_hash: SequenceHash,
    ) -> Option<PoolValue<Block<T, M>>> {
        match self.lookup_map.remove(&sequence_hash) {
            Some(block) => {
                // Remove from priority set
                let priority_key = PriorityKey::new(block.metadata().clone(), sequence_hash);
                self.priority_set.remove(&priority_key);
                Some(block)
            }
            None => None,
        }
    }

    fn match_hashes(
        &mut self,
        hashes: Vec<SequenceHash>,
        return_handle: Arc<ReturnHandleImpl<T, M>>,
    ) -> Vec<PoolItem<Block<T, M>>> {
        let mut matched_blocks = Vec::with_capacity(hashes.len());

        for hash in hashes {
            if let Some(block) = self.take_with_sequence_hash(hash) {
                matched_blocks.push(self.create_pool_item(block, return_handle.clone()));
            } else {
                break;
            }
        }

        let count = matched_blocks.len() as u64;
        self.available_blocks_tx
            .send_modify(|n| *n = n.saturating_sub(count));

        matched_blocks
    }

    fn handle_match_single(&mut self, match_single: MatchSingle<T, M>) {
        let (hash, return_handle, rx) = match_single.dissolve();

        let matched_blocks = self.match_hashes(vec![hash], return_handle);
        let optional_single = matched_blocks.into_iter().next();

        // Send the result back through the channel
        if rx.send(optional_single).is_err() {
            tracing::trace!("Failed to send matched block to requester");
        }
    }

    fn handle_match_multiple(&mut self, match_multiple: MatchMultiple<T, M>) {
        let (hashes, return_handle, rx) = match_multiple.dissolve();

        let matched_blocks = self.match_hashes(hashes, return_handle);

        // Send the matched blocks back through the channel
        if rx.send(matched_blocks).is_err() {
            tracing::trace!("Failed to send matched blocks to requester");
        }
    }

    fn take(&mut self) -> Option<PoolValue<Block<T, M>>> {
        // First try uninitialized blocks - these are often part of sequences
        // that have been arranged in the correct order
        if let Some(block) = self.uninitialized_set.pop_front() {
            return Some(block);
        }

        // if we have blocks in the priority set, pop the first (it's sorted by priority)
        // a fatal error will occur if the block is not found in the lookup map
        if let Some(key) = self.priority_set.pop_first() {
            let block = match self.lookup_map.remove(&key.sequence_hash()) {
                Some(mut block) => {
                    block.reset();
                    block
                }
                None => {
                    panic!("block from priority set not found in lookup map");
                }
            };

            return Some(block);
        }

        None
    }

    fn handle_take(&mut self, take: Take<T, M>) {
        let (count, return_handle, tx) = take.dissolve();

        let mut taken_blocks = Vec::with_capacity(count as usize);

        for _ in 0..count {
            if let Some(block) = self.take() {
                taken_blocks.push(self.create_pool_item(block, return_handle.clone()));
            } else {
                break;
            }
        }

        let count = taken_blocks.len() as u64;
        self.available_blocks_tx
            .send_modify(|n| *n = n.saturating_sub(count));

        // Send the result back through the channel
        if tx.send(taken_blocks).is_err() {
            tracing::trace!("Failed to send matched blocks to requester");
        }
    }

    fn handle_match_request(&mut self, match_request: MatchRequest<T, M>) {
        match match_request {
            MatchRequest::MatchSingle(match_single) => self.handle_match_single(match_single),
            MatchRequest::MatchMultiple(match_multiple) => {
                self.handle_match_multiple(match_multiple)
            }
            MatchRequest::Take(take) => self.handle_take(take),
        }
    }

    fn handle_control_request(&mut self, control_request: ControlRequest<T, M>) {
        match control_request {
            ControlRequest::Insert(insert) => {
                let (block, tx) = insert.dissolve();
                self.handle_insert(block);
                if tx.send(()).is_err() {
                    tracing::trace!("Failed to send insert ack; receiver dropped");
                }
            }
            ControlRequest::UpdateSingle(update_single) => {
                let (update, tx) = update_single.dissolve();
                self.handle_update_single(update);
                if tx.send(()).is_err() {
                    tracing::trace!("Failed to send update single ack; receiver dropped");
                }
            }
            ControlRequest::UpdateMultiple(update_multiple) => {
                let (updates, tx) = update_multiple.dissolve();
                self.handle_update_multiple(updates);
                if tx.send(()).is_err() {
                    tracing::trace!("Failed to send update multiple ack; receiver dropped");
                }
            }
            ControlRequest::Reset(reset) => {
                let (sequence_hashes, tx, _) = reset.dissolve();
                self.handle_reset(sequence_hashes);
                if tx.send(()).is_err() {
                    tracing::trace!("Failed to send reset ack; receiver dropped");
                }
            }
            ControlRequest::ResetAll(reset_all) => {
                let (tx, _) = reset_all.dissolve();
                self.handle_reset_all();
                if tx.send(()).is_err() {
                    tracing::trace!("Failed to send reset all ack; receiver dropped");
                }
            }
        }
    }

    fn handle_insert(&mut self, block: Block<T, M>) {
        self.available_blocks_tx.send_modify(|n| *n += 1);
        self.total_blocks_tx.send_modify(|n| *n += 1);
        self.return_tick += 1;

        self.insert(PoolValue::Direct(block));
    }

    fn handle_return(&mut self, block: PoolValue<Block<T, M>>) {
        self.available_blocks_tx.send_modify(|n| *n += 1);
        self.return_tick += 1;

        self.insert(block);
    }

    fn handle_update_single(&mut self, update: UpdateBlock<M>) {
        self.update_block(vec![update]);
    }

    fn handle_update_multiple(&mut self, updates: Vec<UpdateBlock<M>>) {
        for update in updates {
            if let Some(mut block) = self.take_with_sequence_hash(update.hash) {
                *block.metadata_mut() = update.metadata;
                self.insert(block);
            }
        }
    }

    fn update_block(&mut self, updates: Vec<UpdateBlock<M>>) {
        for update in updates {
            if let Some(mut block) = self.take_with_sequence_hash(update.hash) {
                *block.metadata_mut() = update.metadata;
                self.insert(block);
            }
        }
    }

    fn handle_reset(&mut self, sequence_hashes: Vec<SequenceHash>) {
        for hash in sequence_hashes {
            if let Some(mut block) = self.take_with_sequence_hash(hash) {
                // Reset metadata through deref
                block.metadata_mut().reset_metadata();
                self.insert(block);
            }
        }
    }

    fn handle_reset_all(&mut self) {
        while let Some(priority_key) = self.priority_set.pop_first() {
            if let Some(mut block) = self.lookup_map.remove(&priority_key.sequence_hash()) {
                // reset block -- both state and metadata
                block.reset();
                self.insert(block);
            } else {
                panic!("block from priority set not found in lookup map");
            }
        }
    }
}

impl<T: Storage, M: BlockMetadata> PoolExt<Block<T, M>> for BlockPoolInner<T, M> {}

pub async fn progress_engine<T: Storage + 'static, M: BlockMetadata>(
    match_rx: mpsc::UnboundedReceiver<MatchRequest<T, M>>,
    return_rx: mpsc::UnboundedReceiver<PoolValue<Block<T, M>>>,
    ctrl_rx: mpsc::UnboundedReceiver<ControlRequest<T, M>>,
    fence_rx: mpsc::UnboundedReceiver<oneshot::Sender<()>>,
    mut state: BlockPoolInner<T, M>,
) {
    let mut match_rx = match_rx;
    let mut return_rx = return_rx;
    let mut ctrl_rx = ctrl_rx;
    let mut fence_rx = fence_rx;

    loop {
        tokio::select! {
            biased;

            Some(match_req) = match_rx.recv(), if !match_rx.is_closed() => {
                state.handle_match_request(match_req);
            }

            Some(block) = return_rx.recv(), if !return_rx.is_closed() => {
                state.handle_return(block);
            }

            Some(req) = ctrl_rx.recv(), if !ctrl_rx.is_closed() => {
                state.handle_control_request(req);
            }

            Some(tx) = fence_rx.recv() => {
                if tx.send(()).is_err() {
                    tracing::trace!("Failed to send fence ack; receiver dropped");
                }
            }
        }
    }
}
