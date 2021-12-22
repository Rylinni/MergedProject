# MergedProject

The purpose of this project is replicate the IOS game Merged and build AI that can play the game well.

Rules:
  * If there are three blocks of the same number(1-6) that are neighbors by the four cardinal directions then the block will merge and become a single block that is 1 higher than the original number.
  * If a 6 merges it becomes a "M" block. In the game the simulator this is represented by a 7. Merge "M" blocks and there will be an explosion on the block it merges to.
  * An explosion is when an M block is merged and results in every thing in a 1 block radius being destroyed.
  * You can chain merges in one turn
    * This could be a creating a 2 then a merging the 2 into a 3.
    * You can also perform what I call a "partner" merge. This is when a double block with values 1 and 2, for example, is placed and where the 1 merges with other ones to create a 2 and then that new 2 merges into it's 'partner' 2 to create a 3.
      - caviat-- if the "partner" 2 has enough 2's around it to merge before the 1 block turns into a 2 then the "partner" will disregard the newly created two and merge solely with it's original neighbor 2's. The 1 will still merge into a 2 it just won't merge with the partner


Scoring:
  * You get points for each block that's merged according to the number of the block
  * Multipliers occur when you chain a merge. One chain is a *2 multiplier, two chains is *3  and so on. The multipliers stack.
  * Partner merge-- caviat scoring is different. If you fail to do a proper partner merge then the game will only score your merge of the higher number. - this has not been 100% tested but from what I understand this is the case. I believe its a bug in the game.
