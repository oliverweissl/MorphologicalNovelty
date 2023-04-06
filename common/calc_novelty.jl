using Shuffle
#=
functions:
- Julia version: 
- Author: oliver
- Date: 2023-04-06
=#
INT_CASTER = 10_000
function apply_noise_mask(hist)
    xsize, ysize = size(hist)
    diff = INT_CASTER - sum(hist)

    mask = zeros(Int8, xsize*ysize)
    mask = shuffle(replace(mask, 0=>1, count=diff))
    mask = reshape(mask,(xsize, ysize)) # TODO: if diff > 400 assign 2s and 1s
    return hist + mask
end

function move_supply(supply, fcoord , capacity, tcoord)
    if supply[fcoord] <= capacity[tcoord]
        flow = supply[fcoord]
        capacity[tcoord] -= flow
        supply[fcoord] = 0
    else
        flow = capacity[tcoord]
        supply[fcoord] -= flow
        capacity[tcoord] = 0
    end
    distance = sqrt(abs(fcoord[1]-tcoord[1])^2 + abs(fcoord[2]-tcoord[2])^2)
    score = flow/INT_CASTER*distance
    return score, supply, capacity
end


function wasserstein_distance(hist0::AbstractArray, hist1::AbstractArray)
    xsize, ysize = size(hist0)

    supply, capacity = floor.(Int16, copy(hist0)*INT_CASTER), floor.(Int16, copy(hist1)*INT_CASTER)
    supply, capacity = apply_noise_mask(supply), apply_noise_mask(capacity)

    score = 0
    while true
        from_idx = findfirst(supply .> 0)
        to_idx = findfirst(capacity .> 0)

        if (from_idx == nothing) || (to_idx == nothing)
            break
        end

        work, supply, capacity = move_supply(supply, from_idx, capacity, to_idx)
        score += work
    end
    return score
end

function calculate_novelty(histograms::AbstractArray)
    amt_instances = length(histograms)

    novelty_scores = zeros(amt_instances)
    for i = 1:amt_instances-1
        for j = 1+i:amt_instances
            score = wasserstein_distance(histograms[i], histograms[j])
            novelty_scores[i] += score
            novelty_scores[j] += score
        end
    end
    return novelty_scores
end