# CS 520 Project 2: Aliens - Rematch

## Authors:
Rishik Sarkar & Aditya Girish

## Description:
This repository contains our implementation of grid search strategies used by 8 bots to navigate a 50x50 grid maze, evade aliens, and rescue crew members.

### Ship Label Legend:
- 0: Open Cell
- 1: Closed Cell
- 2: Bot
- 3: Alien
- 4: Crew Member

### Potential Implementation:
- Maintain 50x50 probability matrices (?) containing the probabilities of crew members/aliens being at a each cell
    - Update probabilities each time the sensors are activated
    - Bot picks highest probability neighbor to move to
- Alternative (More efficient):
    - Update only the probabilities of bot's neighbors
    - Move to neighbor with highest crew probability/lowest alien probability

### Alien Probability Thought Process:
- According to Bayes, the probability of an alien being at cell j, given a beep at cell i: P(alien at cell j | beep at cell i) = P(alien at cell j AND beep at cell i) / P(beep at cell i)
    - Let Aj = alien at cell j, B = beep at cell i
    - Thus, we have P(Aj | B) = P(Aj and B) / P(B)
- Let us attempt to determine the probability
    - P(Aj and B) / P(B) 
        - = P(Aj and B) / (Sum of P(A and B) for all possible j values, i.e., all cells)
        - = P(Aj) P(B | Aj) / (Sum of P(A) P(B | A) for all possible j values)
    - We know that P(B | A) = 0 or 1
        - 0 if j value not within sensor range
        - 1 if j value within sensor range
        - This is because the sensor always beeps if an alien is in range
    - Thus, if B occurs:
        - P(Aj | B) = 0 if j not within sensor range
        - P(Aj | B) = P(Aj) / (Sum of P(A) for all possible j values within sensor range)
        - Quite intuitive

### Crew Member Probability Thought Process:
- According to Bayes, the probability of a crew member being at cell i, given a beep: P(crew member at cell i | beep) = P(beep | crew member at cell i) * P(crew member at cell i) / P(beep)
    - Let C = crew member at cell i, B = beep
    - Thus, we have P(C | B) = P(B | C) * P(C) / P(B)
- Let us attempt to determine the probability
    - We know that P(B | C) = e^(-alpha * (d - 1))
    - P(C) = 1 / (total number of open cells - 1), since crew member cannot be at bot's current location
    - P(B) = ...