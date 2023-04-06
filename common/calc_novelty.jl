using Shuffle
#=
functions:
- Julia version: 
- Author: oliver weissl
- Date: 2023-04-06
=#
function apply_noise_mask(hist, INT_CASTER::Int16)
    xsize, ysize = size(hist)
    diff::Int16 = INT_CASTER - sum(hist)

    mask = zeros(Int8, xsize*ysize)
    mask = shuffle(replace(mask, 0=>1, count=diff))
    mask = reshape(mask,(xsize, ysize)) # TODO: if diff > 400 assign 2s and 1s
    return hist + mask
end

function move_supply(supply, fcoord::CartesianIndex , capacity, tcoord::CartesianIndex, INT_CASTER::Int16)
    if supply[fcoord] <= capacity[tcoord]
        flow = supply[fcoord]
        capacity[tcoord] -= flow
        supply[fcoord]::Int8 = 0
    else
        flow = capacity[tcoord]
        supply[fcoord] -= flow
        capacity[tcoord]::Int8 = 0
    end
    distance = sqrt(abs(fcoord[1]-tcoord[1])^2 + abs(fcoord[2]-tcoord[2])^2)
    score = flow/INT_CASTER*distance
    return score, supply, capacity
end

function wasserstein_distance(hist0, hist1, INT_CASTER::Int16)::Float16
    xsize, ysize = size(hist0)

    supply, capacity = floor.(Int16, copy(hist0)*INT_CASTER), floor.(Int16, copy(hist1)*INT_CASTER)
    supply, capacity = apply_noise_mask(supply, INT_CASTER), apply_noise_mask(capacity, INT_CASTER)

    score = 0
    while true
        from_idx = findfirst(supply .> 0)
        to_idx = findfirst(capacity .> 0)

        if (from_idx == nothing) || (to_idx == nothing)
            break
        end

        work, supply, capacity = move_supply(supply, from_idx, capacity, to_idx, INT_CASTER)
        score += work
    end
    return score
end

function calculate_novelty(histograms)
    INT_CASTER::Int16 = 10000
    amt_instances::Int8 = length(histograms) # population = 100 -> int8 goes until 127

    novelty_scores = zeros(amt_instances)
    for i = 1:amt_instances-1
        for j = 1+i:amt_instances
            score = wasserstein_distance(histograms[i], histograms[j], INT_CASTER)
            novelty_scores[i] += score
            novelty_scores[j] += score
        end
    end
    return novelty_scores
end