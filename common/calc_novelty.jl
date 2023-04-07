using Shuffle
#=
functions:
- Julia version: 
- Author: oliver weissl
- Date: 2023-04-06
=#
function find_first_candidate(arr)
    for elem in CartesianIndices(arr)
        if arr[elem] > 0
            return elem
        end
    end
    return nothing
end

function apply_noise_mask(hist, INT_CASTER::UInt)
    diff::UInt = INT_CASTER - sum(hist)

    mask = zeros(UInt, size(hist))
    mask = shuffle(replace(mask, 0=>1, count=diff))
    return hist + mask
end

function move_supply(supply, fcoord::CartesianIndex , capacity, tcoord::CartesianIndex, INT_CASTER::UInt)
    if supply[fcoord] <= capacity[tcoord]
        flow = supply[fcoord]
        capacity[tcoord] -= flow
        supply[fcoord]= 0
    else
        flow = capacity[tcoord]
        supply[fcoord] -= flow
        capacity[tcoord] = 0
    end
    distance = sqrt(abs(fcoord[1]-tcoord[1])^2 + abs(fcoord[2]-tcoord[2])^2)
    score = flow/INT_CASTER*distance
    return score, supply, capacity
end

function wasserstein_distance(hist0, hist1, INT_CASTER::UInt)::Float16
    supply, capacity = trunc.(UInt, hist0.*INT_CASTER), trunc.(UInt, hist1.*INT_CASTER)
    supply, capacity = apply_noise_mask(supply, INT_CASTER), apply_noise_mask(capacity, INT_CASTER)

    score::Float16 = 0
    while true
        from_idx, to_idx = find_first_candidate(supply), find_first_candidate(capacity)

        if (from_idx == nothing) || (to_idx == nothing)
            return score
        end
        work, supply, capacity = move_supply(supply, from_idx, capacity, to_idx, INT_CASTER)
        score += work
    end
end

function calculate_novelty(histograms)
    INT_CASTER::UInt = 10000
    amt_instances::UInt = length(histograms) # population = 100 -> int8 goes until 127

    novelty_scores = zeros(Float16, amt_instances)
    for i = 1:amt_instances-1
        for j = 1+i:amt_instances
            score = wasserstein_distance(histograms[i], histograms[j], INT_CASTER)
            novelty_scores[i] += score
            novelty_scores[j] += score
        end
    end
    return novelty_scores
end

function get_novelties(bricks::Vector{Matrix{Float64}}, hinges::Vector{Matrix{Float64}})
    # ~ 1-1.5 sec
    @time begin
        t1 = Threads.@spawn calculate_novelty(bricks)
        t2 = Threads.@spawn calculate_novelty(hinges)
        y1, y2 = fetch(t1), fetch(t2)
    end
    return y1, y2
end