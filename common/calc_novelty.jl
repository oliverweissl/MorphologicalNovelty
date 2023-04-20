using Shuffle
#=
functions:
- Julia version: 1.8
- Author: oliver weissl
- Date: 2023-04-06
=#
function find_first_candidate(arr)
    @inbounds for elem in CartesianIndices(arr)
        if arr[elem] > 0
            return elem
        end
    end
    return nothing
end

function prepare_histograms!(hist, INT_CASTER)
    hist = floor.(UInt64,hist.*INT_CASTER)
    diff = INT_CASTER - sum(hist)

    mask = zeros(UInt64, size(hist))
    mask[1:diff] .= 1
    shuffle!(mask)
    hist .+= mask
    return hist
end


function move_supply!(supply, capacity, fcoord::CartesianIndex, tcoord::CartesianIndex, INT_CASTER)
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
    return score
end

function wasserstein_distance(hist0, hist1, INT_CASTER)
    supply, capacity = copy(hist0), copy(hist1)  # only use copy when arr of int or float
    score = 0
    while true
        from_idx, to_idx = find_first_candidate(supply), find_first_candidate(capacity)

        if (from_idx == nothing) || (to_idx == nothing)
            return score
        end
        work = move_supply!(supply, capacity, from_idx, to_idx, INT_CASTER)
        score += work
    end
end

function calculate_novelty(histograms::Vector{Matrix{Float64}})
    INT_CASTER::UInt64 = 10000
    amt_instances::UInt64 = length(histograms)

    prepare_histograms!.(histograms, INT_CASTER)
    novelty_scores = zeros(Float64, amt_instances)
    @inbounds for i = 1:amt_instances-1
        @inbounds for j = 1+i:amt_instances
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