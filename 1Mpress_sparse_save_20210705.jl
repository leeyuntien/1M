using CSV, DataFrames, Tables # for manipulating CSV files
using StatsBase # countmap for histograms
using Distributed # distributing for
using DataStructures # PriorityQueue for keeping information gains of partitions in order
using IterTools # subsets for generating combinations of partitions
using Base.Iterators # product for generating products of feature values
using Profile # profiling
using SparseArrays # when data get larger codes tend to be sparser and sparser
using JLD2 # save intermediary results

const MI_BASE = 2 # mutual information base
const MI_MERGE_THRESH = 0.5 # mutual information threshold to merge
const SEC_MERGE_FAIL_PROP = 0.05 # threshold to cut merges
const SEC_COMP_FAILS = 10 # threshold to cut compresses
const MIN_VAL_FREQ = 0 # minimum feature value frequency to consider to compress
const SUB_SAMPLE_DIVISOR = 1 # how many subsamples to keep
const MEMORIZED_LENGTH = 4 # how long partitions to memorize
const SPECIAL_PRIME = 131 # a special prime for combining integer coded feature values
const COMBS_THRESH = 5 # the threshold to use codes or raw

#function main()

    if nprocs() < Sys.CPU_THREADS
        # Distributed.addprocs(Sys.CPU_THREADS - nprocs())
        # Distributed.addprocs(4)
    end

    # @everywhere using LinearAlgebra # dot for getting sumproduct of two array of bits
    using LinearAlgebra # dot for getting sumproduct of two array of bits

    # @everywhere mutable struct CodeDictEntry # code dictionary entry
    mutable struct CodeDictEntry # code dictionary entry
        freq::Int64 # keep sum of code for efficiency
        code::SparseVector{Bool, Int64} # corresponding code in data
    end

    # @everywhere mutable struct DataDictEntry # data dictionary entry
    mutable struct DataDictEntry # data dictionary entry
        entropy::Float64 # keep entropy of fields for efficiency
        codes::Array{BitArray} # all codes related to the fields
    end

    # @everywhere mutable struct PriorityQueueEntry
    mutable struct PriorityQueueEntry
        features::Tuple{Array{String, 1}, Array{String, 1}}
        priority::Int64
    end

    # @everywhere function ent(f, base=MI_BASE)::Float64
    function ent(f, base=MI_BASE)::Float64
        # if !iszero(x)
        #     return -x / n * log(base, x / n)
        # else
        #     return 0.0
        # end
        ifelse(f > 0.0, -f * log(base, f), 0.0)
    end

    function toBigInt(x)
        num::Int128 = 1
        for item in x
            num += item
            num *= SPECIAL_PRIME # a large enough prime
        end
        return num
    end

    # @everywhere function overlappedEnt(codes_a, codes_b, ent_base)
    function overlappedEnt(codes_a, codes_b, ent_base)
        total_ent = 0.0
        # for code_a in codes_a, code_b in codes_b
            # overlapped_freq = dot(code_a, code_b)
            # if overlapped_freq > 0
            #     total_ent += ent(overlapped_freq / n)
            # end
            # total_ent += ent(dot(code_a, code_b) / ent_base)
        # end
        # return sum(ent(dot(code_a, code_b) / ent_base)::Float64 for code_a in codes_a, code_b in codes_b)
        for code_a in codes_a, code_b in codes_b
            total_count = LinearAlgebra.dot(code_a, code_b)
            total_ent += ent(total_count / ent_base)
        end
        return total_ent
    end

    function overlappedEnt(data_raw_array, data_raw_names, data_raw_code_dict, partition_list, part, ent_base)
        partition_list_array = map(x -> findfirst(isequal(x), data_raw_names), partition_list)
        part_array = map(x -> findfirst(isequal(x), data_raw_names), part)
        total_ent_partition_list = 0.0
        if haskey(data_raw_code_dict, partition_list)
            total_ent_partition_list = data_raw_code_dict[partition_list].entropy
        else
            # total_ent_partition_list = sum(ent.(values(countmap(collect(eachrow(data_raw[partition_list])))) ./ ent_base))
            total_ent_partition_list = sum(ent.(values(countmap(map(x -> toBigInt(x), [r[:] for r in eachrow(data_raw_array[:, partition_list_array])]))) ./ ent_base))
        end
        total_ent_part = 0.0
        if haskey(data_raw_code_dict, part)
            total_ent_part = data_raw_code_dict[part].entropy
        else
            # total_ent_part = sum(ent.(values(countmap(collect(eachrow(data_raw[part])))) ./ ent_base))
            total_ent_part = sum(ent.(values(countmap(map(x -> toBigInt(x), [r[:] for r in eachrow(data_raw_array[:, part_array])]))) ./ ent_base))
        end
        total_ent_partition_list_part = 0.0
        if haskey(data_raw_code_dict, vcat(partition_list, part))
            total_ent_partition_list_part = data_raw_code_dict[vcat(partition_list, part)].entropy
        elseif haskey(data_raw_code_dict, partition_list) && haskey(data_raw_code_dict, part)
            if length(data_raw_code_dict[partition_list].codes) <= COMBS_THRESH && length(data_raw_code_dict[part].codes) <= COMBS_THRESH
                total_ent_partition_list_part = overlappedEnt(data_raw_code_dict[partition_list].codes, data_raw_code_dict[part].codes, n ÷ SUB_SAMPLE_DIVISOR)
            else
                total_ent_partition_list_part = sum(ent.(values(countmap(map(x -> toBigInt(x), [r[:] for r in eachrow(data_raw_array[:, vcat(partition_list_array, part_array)])]))) ./ ent_base))
            end
        else
            # total_ent_partition_list_part = sum(ent.(values(countmap(collect(eachrow(data_raw[vcat(partition_list, part)])))) ./ ent_base))
            total_ent_partition_list_part = sum(ent.(values(countmap(map(x -> toBigInt(x), [r[:] for r in eachrow(data_raw_array[:, vcat(partition_list_array, part_array)])]))) ./ ent_base))
        end
        return total_ent_partition_list_part - total_ent_partition_list - total_ent_part
    end

    data_path = "C:/Users/Yuntien.Lee/Documents/Work/Data/code/Med/GEHA/"
    data_file = "services_by_claim_id_int_coded_20201021_100000.csv"
    data_raw = CSV.File(string(data_path, data_file), header=true) |> DataFrame # string( , ) to concatenate two strings
    selecata_raw, Not(Symbol(80))) # :TC_MAIN_NAME too many possible values
    for amt_field in vcat([14, 16, 28], collect(17:26)) # No :AMT fields
        select!(data_raw, Not(Symbol(amt_field)))
    end
    # @everywhere n, m = $10000001, $200
    n, m = size(data_raw)
    # @everywhere MI_BASE = $MI_BASE
    println("number of records ", n, ", number of fields ", m)

@time begin
    partition_code_dict = Dict{Array{String, 1}, Dict{Array{Int64, 1}, CodeDictEntry}}() # ((F0), (F1)) -> (F0_0, F1_0) -> (count, cost, code)
    data_raw_code_dict = Dict{Array{String, 1}, DataDictEntry}() # ((F0), (F1)) -> (cost, array of codes)
    pair_mi_dict = Dict{Array{String, 1}, Float64}() # A lookup table for all mi between pairs
    subsample_indices = StatsBase.sample(1:n, n ÷ SUB_SAMPLE_DIVISOR)
    for field::String in string.(names(data_raw)) # loop through all fields
        partition_code_dict[[field]] = Dict{Array{Int64, 1}, CodeDictEntry}()
        data_raw_code_dict[[field]] = DataDictEntry(0.0, Array{BitArray, 1}()) # DataDictEntry(entropy, codes)
        value_freq_dict = StatsBase.countmap(data_raw[:, Symbol(field)]) # F0_1 -> 2; F0_2 -> 4, ...
        for (value::Int64, freq::Int64) in value_freq_dict
            partition_code_dict[[field]][[value]] = CodeDictEntry(freq, sparse(data_raw[:, Symbol(field)] .== value))
        end
        value_freq_dict = StatsBase.countmap(data_raw[:, Symbol(field)])
        sum_ent = 0.0
        for (value::Int64, freq::Int64) in value_freq_dict
            push!(data_raw_code_dict[[field]].codes, data_raw[:, Symbol(field)] .== value)
            sum_ent += ent(freq / (n ÷ SUB_SAMPLE_DIVISOR))
        end
        data_raw_code_dict[[field]].entropy = sum_ent
        println("code dict and data dict on ", field, " finished")
    end

    # free memory
    data_raw_names = string.(names(data_raw))
    data_raw_array = data_raw |> Tables.matrix
    data_raw = nothing
    GC.gc()

    # define dictionaries to make (part_a, part_b) loops parallel
    #=@everywhere rd_pair(part_a::Tuple{Array{Symbol, 1}, Array{Symbol, 1}, Array{BitArray, 1}, Array{BitArray, 1}, Int64}, part_b::Tuple{Array{Symbol, 1}, Array{Symbol, 1}, Array{BitArray, 1}, Array{BitArray, 1}, Int64}) = rd_pair(rd_pair(Dict{Tuple{Array{Symbol, 1}, Array{Symbol, 1}}, Float64}(), part_a), part_b)
    @everywhere rd_pair(dict::Dict{Tuple{Array{Symbol, 1}, Array{Symbol, 1}}, Float64}, part::Tuple{Array{Symbol, 1}, Array{Symbol, 1}, Array{BitArray, 1}, Array{BitArray, 1}, Int64}) =
        begin
            dict[(part[1], part[2])] = overlappedEnt(part[3], part[4], part[5]);
            dict
        end
    @everywhere rd_pair(dict_a::Dict{Tuple{Array{Symbol, 1}, Array{Symbol, 1}}, Float64}, dict_b::Dict{Tuple{Array{Symbol, 1}, Array{Symbol, 1}}, Float64}) = merge!(dict_a, dict_b)

    # define dictionaries to make (code_a, code_b) loops parallel
    @everywhere rd_code_pair(part_a::Tuple{Array{Symbol, 1}, Array{Symbol, 1}, BitArray, BitArray, Int64}, part_b::Tuple{Array{Symbol, 1}, Array{Symbol, 1}, BitArray, BitArray, Int64}) = rd_code_pair(rd_code_pair(Dict{Tuple{Array{Symbol, 1}, Array{Symbol, 1}}, Float64}(), part_a), part_b)
    @everywhere rd_code_pair(dict::Dict{Tuple{Array{Symbol, 1}, Array{Symbol, 1}}, Float64}, part::Tuple{Array{Symbol, 1}, Array{Symbol, 1}, BitArray, BitArray, Int64}) =
        begin
            total_count = LinearAlgebra.dot(part[3], part[4])
            dict[(part[1], part[2])] = get(dict, (part[1], part[2]), 0) + ent(total_count / part[5]);
            dict
        end
    @everywhere rd_code_pair(dict_a::Dict{Tuple{Array{Symbol, 1}, Array{Symbol, 1}}, Float64}, dict_b::Dict{Tuple{Array{Symbol, 1}, Array{Symbol, 1}}, Float64}) = merge!(+, dict_a, dict_b)

    # define arrays to make (part_a, part_b) loops parallel
    @everywhere ra_pair(part_a::Tuple, part_b::Tuple) = ra_pair(ra_pair(Vector(), part_a), part_b)
    @everywhere ra_pair(arr::Vector, part::Tuple) =
        begin push!(arr,
            ((part[1], part[2]), overlappedEnt(part[3], part[4], part[5])));
            arr
        end
    @everywhere ra_pair(arr_a::Vector, arr_b::Vector) = begin arr_a = vcat(arr_a, arr_b) end=#

    partition_ig_pq = DataStructures.PriorityQueue{Tuple{Array{String, 1}, Array{String, 1}}, Float64}()
    #=to_distribute = []
    for (partition_a::Symbol, partition_b::Symbol) in IterTools.subsets([Symbol("F$j") for j in 0:199], 2) # maybe DArray
        for code_a in data_raw_code_dict[[partition_a]].codes, code_b in data_raw_code_dict[[partition_b]].codes
            push!(to_distribute, ([partition_a], [partition_b], code_a, code_b))
        end
    end
    overlapped_ent_to_add = @distributed (rd_code_pair) for comp in to_distribute
        (comp[1], comp[2], comp[3], comp[4], n ÷ SUB_SAMPLE_DIVISOR)
    end
    for (part::Tuple{Array{Symbol, 1}, Array{Symbol, 1}}, overlapped_ent::Float64) in overlapped_ent_to_add # parallel dict
        ig::Float64 = (data_raw_code_dict[part[1]].entropy + data_raw_code_dict[part[2]].entropy - overlapped_ent) / 2
        enqueue!(partition_ig_pq, (part[1], part[2]), -ig) # min heap so take most negative
        println("pushed ", (part[1], part[2]), ' ', -ig, " entry to priority queue for merging")
    end=#
    for (partition_a::String, partition_b::String) in IterTools.subsets(data_raw_names, 2) # maybe DArray
        # information gain = entropy of first partition + entropy of second partition - entropy of overlapped part
        ig::Float64 = (data_raw_code_dict[[partition_a]].entropy + data_raw_code_dict[[partition_b]].entropy - overlappedEnt(data_raw_code_dict[[partition_a]].codes, data_raw_code_dict[[partition_b]].codes, n ÷ SUB_SAMPLE_DIVISOR)) / 2
        enqueue!(partition_ig_pq, ([partition_a], [partition_b]), -ig) # min heap so take most negative
        pair_mi_dict[[partition_a, partition_b]] = ig
        pair_mi_dict[[partition_b, partition_a]] = ig
        println("pushed ", ([partition_a], [partition_b]), ' ', -ig, " entry to priority queue for merging")
    end

    failsAfterSuccMerge = 0
    secMergeFailThresh = trunc(Int, length(partition_ig_pq) * SEC_MERGE_FAIL_PROP) + 1
    current_partition_step = 0 # keep track of number of steps taken in partition loop
    current_partition_pair = Tuple{Array{String, 1}, Array{String, 1}}(([], []))
    current_partition_pair_code_dict = Dict()
    current_partition_pair_code_dict_best = Dict()
    unique_value_pq = Array{Tuple{Tuple{Array{Int64, 1}, Array{Int64, 1}}, Int64}, 1}() # PriorityQueue{Tuple{Array{String, 1}, Array{String, 1}}, Int64}()
    overlapped_code = sparse(zeros(Bool, n))
    @load ("1M_" * ARGS[1] * ".jld2") current_partition_step failsAfterSuccMerge secMergeFailThresh partition_ig_pq partition_code_dict data_raw_code_dict pair_mi_dict
    while length(partition_ig_pq) > 0 && failsAfterSuccMerge < secMergeFailThresh
        (partition_pair::Tuple{Array{String, 1}, Array{String, 1}}, current_ig::Float64) = DataStructures.peek(partition_ig_pq)
        dequeue!(partition_ig_pq)
        # @everywhere current_partition_pair = $partition_pair
        global current_partition_pair = partition_pair
        println("current partition pair ", partition_pair, ' ', current_ig)
        # total_cost = current total cost
        # singleton_freq_dict = singleton frequency dictionary to update
        # total_freq = total count of code
        # singleton_cost = singletone cost
        # total_cost, singleton_freq_dict, total_freq, singleton_cost = encoding_cost(partition_code_dict)
        encoded_data_cost = 0.0
        pattern_cost = 0.0 # think
        freqs = Array{Int64, 1}() # array of freqency in all partitions and all feature values
        for (part, code_dict) in partition_code_dict
            for (value, code_entry) in code_dict
                push!(freqs, code_entry.freq)
            end
        end
        total_freq = sum(freqs)
        pattern_cost -= sum(log.(MI_BASE, freqs / total_freq))
        encoded_data_cost -= sum(freqs .* log.(MI_BASE, freqs / total_freq))
        singleton_freq_dict = Dict{Int64, Int64}() # think
        for (part, code_dict) in partition_code_dict
            for value in keys(code_dict)
                # merge!(+, singleton_freq_dict, countmap(flattenFeatureValues(value)))
                merge!(+, singleton_freq_dict, countmap(value))
            end
        end
        singleton_freqs::Array{Int64, 1} = collect(values(singleton_freq_dict))
        total_singleton_freq::Int64 = sum(singleton_freqs)
        singleton_cost::Float64 = -sum(singleton_freqs .* log.(MI_BASE, singleton_freqs / total_singleton_freq))
        total_cost::Float64 = encoded_data_cost + singleton_cost + pattern_cost
        global current_partition_step += 1
        println("current partition step ", current_partition_step, ' ', total_cost)
        # initialize current code table, need to sort by (length, cost)
        global current_partition_pair_code_dict = deepcopy(merge(partition_code_dict[current_partition_pair[1]], partition_code_dict[current_partition_pair[2]]))
        global current_partition_pair_code_dict_best = deepcopy(current_partition_pair_code_dict)
        overlapped_freq = 0
        value_a_freq = 0
        value_b_freq = 0
        global unique_value_pq = Array{Tuple{Tuple{Array{Int64, 1}, Array{Int64, 1}}, Int64}, 1}() # PriorityQueue{Tuple{Array{String, 1}, Array{String, 1}}, Int64}()
        for elem_a in partition_code_dict[current_partition_pair[1]], elem_b in partition_code_dict[current_partition_pair[2]]
            overlapped_freq = dot(elem_a[2].code, elem_b[2].code)
            if overlapped_freq > 0
                push!(unique_value_pq, ((elem_a[1], elem_b[1]), overlapped_freq)) # enqueue!(unique_value_pq, (elem_a[1], elem_b[1]), -overlapped_freq)
            end
        end
        # sort!(unique_value_pq, by = x -> x[2] * 100000 + sum(map(x -> parse(Int64, x[end:end]), x[1][1])) * 1000 + sum(map(x -> parse(Int64, x[end:end]), x[1][2])), rev = true) # assume length of values does not exceed 100000
        sort!(unique_value_pq, by = x -> x[2] * 100000 + length(x[1][1]) + length(x[1][2]), rev = true) # assume length of values does not exceed 100000
        total_cost_updated = total_cost
        mean_reduced_cost = Array{Float64, 1}() # keep track of the threshold to determine whether to merge
        failsAfterSuccComp = 0
        num_features_nonfocus = sum([length(code_dict) for (part, code_dict) in partition_code_dict if !((part ⊆ current_partition_pair[1]) || (part ⊆ current_partition_pair[2]))])
        while length(unique_value_pq) > 0 && failsAfterSuccComp < SEC_COMP_FAILS
            value_pair = popfirst!(unique_value_pq)[1] # dequeue!(unique_value_pq)
            if haskey(current_partition_pair_code_dict, value_pair[1]) && haskey(current_partition_pair_code_dict, value_pair[2])
                global overlapped_code = current_partition_pair_code_dict[value_pair[1]].code .& current_partition_pair_code_dict[value_pair[2]].code
                overlapped_freq = sum(overlapped_code)
                value_a_freq = current_partition_pair_code_dict[value_pair[1]].freq
                value_b_freq = current_partition_pair_code_dict[value_pair[2]].freq
            else
                println("key not exist ", value_pair[1], ' ', value_pair[2])
                global overlapped_code = sparse(zeros(Bool, n))
                overlapped_freq = 0
                value_a_freq = 0
                value_b_freq = 0
            end
            if overlapped_freq > MIN_VAL_FREQ # 0
                # deal with the first feature value in the pair
                diff_code::SparseVector{Bool, Int64} = current_partition_pair_code_dict[value_pair[1]].code .⊻ overlapped_code # take XOR to get difference in code
                diff_freq = sum(diff_code) # take sum to get frequency
                if diff_freq != (current_partition_pair_code_dict[value_pair[1]].freq - overlapped_freq) # should match
                    println("diff_code not match diff_count", value_pair[1])
                    del_a = false
                elseif diff_freq > 0 # edit original entry
                    global current_partition_pair_code_dict[value_pair[1]].freq = diff_freq
                    global current_partition_pair_code_dict[value_pair[1]].code = diff_code
                    del_a = false
                else # delete this entry
                    del_a = true
                end
                # deal with the second feature value in the pair
                diff_code = current_partition_pair_code_dict[value_pair[2]].code .⊻ overlapped_code # take XOR to get difference in code
                diff_freq = sum(diff_code) # take sum to get frequency
                if diff_freq != (current_partition_pair_code_dict[value_pair[2]].freq - overlapped_freq) # should match
                    println("diff_code not match diff_count", value_pair[2])
                    del_b = false
                elseif diff_freq > 0 # edit original entry
                    global current_partition_pair_code_dict[value_pair[2]].freq = diff_freq
                    global current_partition_pair_code_dict[value_pair[2]].code = diff_code
                    del_b = false
                else # delete this entry
                    del_b = true
                end
                total_freq -= overlapped_freq # take out overlapped freq
                diff_cost = 0.0
                if del_a && del_b # delete both entries
                    diff_cost = 2 * log(MI_BASE, overlapped_freq / (total_freq + overlapped_freq))
                    delete!(current_partition_pair_code_dict, value_pair[1])
                    delete!(current_partition_pair_code_dict, value_pair[2])
                    total_cost_updated += overlapped_freq * diff_cost # cost update for both
                elseif del_a # delete the first entry
                    diff_cost = log(MI_BASE, (current_partition_pair_code_dict[value_pair[2]].freq + overlapped_freq) / current_partition_pair_code_dict[value_pair[2]].freq) + log(MI_BASE, overlapped_freq / (total_freq + overlapped_freq))
                    delete!(current_partition_pair_code_dict, value_pair[1])
                    total_cost_updated += overlapped_freq * log(MI_BASE, overlapped_freq / (total_freq + overlapped_freq)) # cost update for the first entry
                    total_cost_updated += (current_partition_pair_code_dict[value_pair[2]].freq + overlapped_freq) * log(MI_BASE, (current_partition_pair_code_dict[value_pair[2]].freq + overlapped_freq) / (total_freq + overlapped_freq)) # cost update for the original second entry
                    total_cost_updated -= current_partition_pair_code_dict[value_pair[2]].freq * log(MI_BASE, current_partition_pair_code_dict[value_pair[2]].freq / (total_freq + overlapped_freq)) # cost update for the updated second entry
                elseif del_b # delete the second entry
                    diff_cost = log(MI_BASE, (current_partition_pair_code_dict[value_pair[1]].freq + overlapped_freq) / current_partition_pair_code_dict[value_pair[1]].freq) + log(MI_BASE, overlapped_freq / (total_freq + overlapped_freq))
                    delete!(current_partition_pair_code_dict, value_pair[2])
                    total_cost_updated += overlapped_freq * log(MI_BASE, overlapped_freq / (total_freq + overlapped_freq)) # cost update for the second entry
                    total_cost_updated += (current_partition_pair_code_dict[value_pair[1]].freq + overlapped_freq) * log(MI_BASE, (current_partition_pair_code_dict[value_pair[1]].freq + overlapped_freq) / (total_freq + overlapped_freq)) # cost update for the original first entry
                    total_cost_updated -= current_partition_pair_code_dict[value_pair[1]].freq * log(MI_BASE, current_partition_pair_code_dict[value_pair[1]].freq / (total_freq + overlapped_freq)) # cost update for the updated first entry
                else # no deletion
                    diff_cost = log(MI_BASE, (current_partition_pair_code_dict[value_pair[1]].freq + overlapped_freq) / current_partition_pair_code_dict[value_pair[1]].freq) + log(MI_BASE, (current_partition_pair_code_dict[value_pair[2]].freq + overlapped_freq) / current_partition_pair_code_dict[value_pair[2]].freq)
                    total_cost_updated += (current_partition_pair_code_dict[value_pair[1]].freq + overlapped_freq) * log(MI_BASE, (current_partition_pair_code_dict[value_pair[1]].freq + overlapped_freq) / (total_freq + overlapped_freq)) + (current_partition_pair_code_dict[value_pair[2]].freq + overlapped_freq) * log(MI_BASE, (current_partition_pair_code_dict[value_pair[2]].freq + overlapped_freq) / (total_freq + overlapped_freq)) # cost update for the original entries
                    total_cost_updated -= current_partition_pair_code_dict[value_pair[1]].freq * log(MI_BASE, current_partition_pair_code_dict[value_pair[1]].freq / (total_freq + overlapped_freq)) + current_partition_pair_code_dict[value_pair[2]].freq * log(MI_BASE, current_partition_pair_code_dict[value_pair[2]].freq / (total_freq + overlapped_freq)) # cost update for the updated entries
                end
                total_cost_updated += diff_cost
                diff_cost = log(MI_BASE, (total_freq + overlapped_freq) / total_freq)
                # other_feature_values = sum(length(keys(code_dict))
                #                             for (part, code_dict) in partition_code_dict
                #                             if !(part in current_partition_pair))
                total_cost_updated -= (num_features_nonfocus + length(current_partition_pair_code_dict)) * diff_cost
                total_cost_updated -= (total_freq - overlapped_freq) * diff_cost
                #= for (value, code_entry) in current_partition_pair_code_dict # all existing entries reduce the same cost
                    current_partition_pair_code_dict[value] = (code_entry[1], code_entry[2] - diff_cost, code_entry[3])
                end =#
                diff_cost = -log(MI_BASE, overlapped_freq / total_freq)
                total_cost_updated += diff_cost
                total_cost_updated += overlapped_freq * diff_cost
                global current_partition_pair_code_dict[vcat(value_pair[1], value_pair[2])] = CodeDictEntry(overlapped_freq, overlapped_code) # insert into current_partition_pair_code_dict
                if !del_a && !del_b
                    merge!(+, singleton_freq_dict, StatsBase.countmap(vcat(value_pair[1], value_pair[2])))
                elseif !del_b
                    merge!(+, singleton_freq_dict, StatsBase.countmap(value_pair[2]))
                elseif !del_a
                    merge!(+, singleton_freq_dict, StatsBase.countmap(value_pair[1]))
                end
                if !del_a || !del_b
                    singleton_freqs = collect(values(singleton_freq_dict))
                    total_singleton_freq = sum(singleton_freqs)
                    total_cost_updated -= singleton_cost + sum(singleton_freqs .* log.(MI_BASE, singleton_freqs / total_singleton_freq))
                    singleton_cost = -sum(singleton_freqs .* log.(MI_BASE, singleton_freqs / total_singleton_freq))
                end
                if total_cost > total_cost_updated
                    failsAfterSuccComp = 0
                    push!(mean_reduced_cost, (total_cost - total_cost_updated) / (value_a_freq + value_b_freq - overlapped_freq))
                    total_cost = total_cost_updated
                    global current_partition_pair_code_dict_best[vcat(value_pair[1], value_pair[2])] = CodeDictEntry(overlapped_freq, overlapped_code)
                    if del_a && del_b
                        delete!(current_partition_pair_code_dict_best, value_pair[1])
                        delete!(current_partition_pair_code_dict_best, value_pair[2])
                    elseif del_a
                        delete!(current_partition_pair_code_dict_best, value_pair[1])
                        global current_partition_pair_code_dict_best[value_pair[2]].freq = current_partition_pair_code_dict[value_pair[2]].freq
                        global current_partition_pair_code_dict_best[value_pair[2]].code = current_partition_pair_code_dict[value_pair[2]].code
                    elseif del_b
                        delete!(current_partition_pair_code_dict_best, value_pair[2])
                        global current_partition_pair_code_dict_best[value_pair[1]].freq = current_partition_pair_code_dict[value_pair[1]].freq
                        global current_partition_pair_code_dict_best[value_pair[1]].code = current_partition_pair_code_dict[value_pair[1]].code
                    else
                        global current_partition_pair_code_dict_best[value_pair[1]].freq = current_partition_pair_code_dict[value_pair[1]].freq
                        global current_partition_pair_code_dict_best[value_pair[1]].code = current_partition_pair_code_dict[value_pair[1]].code
                        global current_partition_pair_code_dict_best[value_pair[2]].freq = current_partition_pair_code_dict[value_pair[2]].freq
                        global current_partition_pair_code_dict_best[value_pair[2]].code = current_partition_pair_code_dict[value_pair[2]].code
                    end
                else
                    failsAfterSuccComp += 1
                    total_freq += overlapped_freq
                    total_cost_updated = total_cost
                    # not necessary
                    if haskey(current_partition_pair_code_dict, vcat(value_pair[1], value_pair[2]))
                        delete!(current_partition_pair_code_dict, vcat(value_pair[1], value_pair[2]))
                    end
                    if del_a && del_b # think
                        global current_partition_pair_code_dict[value_pair[1]] = CodeDictEntry(overlapped_freq, overlapped_code)
                        global current_partition_pair_code_dict[value_pair[2]] = CodeDictEntry(overlapped_freq, overlapped_code)
                    elseif del_a
                        global current_partition_pair_code_dict[value_pair[1]] = CodeDictEntry(overlapped_freq, overlapped_code)
                        global current_partition_pair_code_dict[value_pair[2]].freq += overlapped_freq
                        global current_partition_pair_code_dict[value_pair[2]].code .|= overlapped_code
                        merge!(-, singleton_freq_dict, StatsBase.countmap(value_pair[2]))
                    elseif del_b
                        global current_partition_pair_code_dict[value_pair[2]] = CodeDictEntry(overlapped_freq, overlapped_code)
                        global current_partition_pair_code_dict[value_pair[1]].freq += overlapped_freq
                        global current_partition_pair_code_dict[value_pair[1]].code .|= overlapped_code
                        merge!(-, singleton_freq_dict, StatsBase.countmap(value_pair[1]))
                    else
                        global current_partition_pair_code_dict[value_pair[1]].freq += overlapped_freq
                        global current_partition_pair_code_dict[value_pair[1]].code .|= overlapped_code
                        global current_partition_pair_code_dict[value_pair[2]].freq += overlapped_freq
                        global current_partition_pair_code_dict[value_pair[2]].code .|= overlapped_code
                        merge!(-, singleton_freq_dict, StatsBase.countmap(vcat(value_pair[1], value_pair[2])))
                    end
                    diff_cost = log(MI_BASE, (total_freq - overlapped_freq) / total_freq)
                    #= for (value, code_entry) in current_partition_pair_code_dict # all existing entries reduce the same cost
                        current_partition_pair_code_dict[value] = (code_entry[1], code_entry[2] - diff_cost, code_entry[3])
                    end =#
                    singleton_freqs = collect(values(singleton_freq_dict))
                    total_singleton_freq = sum(singleton_freqs)
                    singleton_cost = -sum(singleton_freqs .* log.(MI_BASE, singleton_freqs / total_singleton_freq))
                end
            else
                break
            end
        end
        current_partition_pair_length::Int64 = length(current_partition_pair[1]) + length(current_partition_pair[2])
        mi::Float64 = current_partition_pair_length * current_ig
        if length(mean_reduced_cost) > 0 && sum(mean_reduced_cost) / length(mean_reduced_cost) > -1 / mi
            global failsAfterSuccMerge = 0
            # sort dictionary according to (length, entropy)
            # current_partition_pair_code_dict_best = Dict(sort(collect(current_partition_pair_code_dict_best), by = x -> (length(flattenFeatureValues(x[1])), x[2]), rev = true))
            # partition_code_dict[current_partition_pair] = copy(current_partition_pair_code_dict_best)
            partition_list = vcat(current_partition_pair[1], current_partition_pair[2])
            partition_code_dict[partition_list] = current_partition_pair_code_dict_best
            delete!(partition_code_dict, current_partition_pair[1])
            delete!(partition_code_dict, current_partition_pair[2])
            # remove items in partition_ig_pq
            # not working since changing order filter!(elem -> !(elem[1] in current_partition_pair) && !(elem[2] in current_partition_pair), partition_ig_pq)
            for (part::Tuple{Array{String, 1}, Array{String, 1}}, ig::Float64) in partition_ig_pq
                if (part[1] ⊆ current_partition_pair[1]) || (part[1] ⊆ current_partition_pair[2]) || (part[2] ⊆ current_partition_pair[1]) || (part[2] ⊆ current_partition_pair[2])
                    delete!(partition_ig_pq, part)
                end
            end
            if length(partition_list) <= MEMORIZED_LENGTH
                data_raw_code_dict[partition_list] = DataDictEntry(0.0, Array{BitArray, 1}())
                total_ent = 0.0
                for code_a in data_raw_code_dict[current_partition_pair[1]].codes
                    for code_b in data_raw_code_dict[current_partition_pair[2]].codes
                        overlapped_code::BitArray = code_a .& code_b
                        overlapped_freq = mapreduce(count_ones, +, overlapped_code.chunks)
                        if overlapped_freq > 0
                            push!(data_raw_code_dict[partition_list].codes, overlapped_code)
                            total_ent += ent(overlapped_freq / (n ÷ SUB_SAMPLE_DIVISOR))
                        end
                    end
                end
                data_raw_code_dict[partition_list].entropy = total_ent
            end
@time begin
            # all_other_parts::Array{Array{Symbol, 1}, 1} = filter(elem -> elem != partition_list, collect(keys(partition_code_dict))) # map(elem -> (current_partition_pair, elem), filter(elem -> elem != current_partition_pair, collect(keys(partition_code_dict))))
            for part in keys(partition_code_dict)
                if part != partition_list
                    mi_avg_intra_current = 0
                    for (x, y) in IterTools.subsets(partition_list, 2)
                        mi_avg_intra_current += get(pair_mi_dict, [x, y], pair_mi_dict[[y, x]])
                    end
                    mi_avg_inter = 0
                    for (x, y) in IterTools.product(partition_list, part)
                        mi_avg_inter += get(pair_mi_dict, [x, y], pair_mi_dict[[y, x]])
                    end
                    if MI_MERGE_THRESH * mi_avg_intra_current / (current_partition_pair_length * (current_partition_pair_length - 1) / 2) < mi_avg_inter / (length(partition_list) * length(part)) * current_partition_pair_length * length(part)
                        enqueue!(partition_ig_pq, (partition_list, part), 
                            overlappedEnt(data_raw_array, data_raw_names, data_raw_code_dict, partition_list, part, n ÷ SUB_SAMPLE_DIVISOR) / (current_partition_pair_length + length(part)))
                    end
                end
            end
end
        else
            global failsAfterSuccMerge += 1
        end
        global secMergeFailThresh = trunc(Int, length(partition_ig_pq) * SEC_MERGE_FAIL_PROP) + 1
        #=if parse(Int64, ARGS[1]) == current_partition_step
            @save ("1M_" * ARGS[1] * ".jld2") current_partition_step failsAfterSuccMerge secMergeFailThresh partition_ig_pq partition_code_dict data_raw_code_dict pair_mi_dict
        end=#
    end
end

    # get final scores
    code_matrix = Array{SparseVector{Bool, Int64}, 1}()
    # code_matrix = Array{BitArray, 1}()
    for (part, code_dict) in partition_code_dict
        for (value, code_entry) in code_dict
            push!(code_matrix, code_entry.code)
        end
    end
    println("size of code_matrix ", length(code_matrix))
    @show keys(partition_code_dict)
    @time begin
        freqs = map(sum, code_matrix)
        total_freq = sum(freqs)
        pattern_costs = -log.(MI_BASE, freqs ./ total_freq)
        intrinsic_scores = hcat(code_matrix...) * pattern_costs
    end
    @save "/mnt/data/scores_142792177.jld2" intrinsic_scores


function overlapped_code_ent(code_list_a, code_list_b, n)
    total = 0.0
    for code_a in data_raw_code_dict[code_list_a][2]
        for code_b in data_raw_code_dict[code_list_b][2]
            # overlapped_freq = dot(code_a, code_b)
            # if overlapped_freq > 0
            #     total += ent(overlapped_freq / n)
            # end
            total += ent(dot(code_a, code_b) / n)
        end
    end
    println(total)
    return total
end

function test_a()
@time begin
np = nprocs()
if (np == 1)
    addprocs(Sys.CPU_THREADS - 1)
end
np = nprocs()
current_partition_pair_length = length(flattenPartitionPairs(current_partition_pair))
current_partition_pair_ent = data_raw_code_dict[current_partition_pair][1]
parts = collect(keys(partition_code_dict))
results = Dict()
@sync begin
           for p = 1:np
               if p != myid() || np == 1
                   @async begin
                       for part in parts
                           part_ent = data_raw_code_dict[part][1]
                           overlapped_ent = remotecall_fetch(overlapped_code_ent, p, data_raw_code_dict, current_partition_pair, part, 2, n)
                           ig = (current_partition_pair_ent + part_ent - overlapped_ent) / (current_partition_pair_length + length(flattenPartitionPairs(part)))
                           results[part] = ig
                       end
                   end
               end
           end
       end
end
end

function test_b()
@time begin
parts = collect(keys(partition_code_dict))
results = zeros(length(parts))
for idx in 1:length(parts)
    results[idx] = overlapped_code_ent(parts[1], parts[idx], 1000000)
end
end
end

function test_c()
@time begin
current_partition_pair_length = length(flattenPartitionPairs(current_partition_pair))
println(current_partition_pair_length)
parts = collect(keys(partition_code_dict))
current_partition_pair_ent = data_raw_code_dict[current_partition_pair][1]
results = Dict()
oe = Dict()
@time begin
    for part in parts
    part_ent = data_raw_code_dict[part][1]
    overlapped_ent = overlapped_code_ent(data_raw_code_dict, current_partition_pair, part, 2, n)
    ig = (current_partition_pair_ent + part_ent - overlapped_ent) / (current_partition_pair_length + length(flattenPartitionPairs(part)))
    results[part] = ig
    oe[part] = overlapped_ent
    end
end
end
end

function test_d()
    current_partition_pair_length = length(flattenPartitionPairs(current_partition_pair))
    println(current_partition_pair_length)
    parts = collect(keys(partition_code_dict))
    current_partition_pair_ent = data_raw_code_dict[current_partition_pair][1]
    @everywhere rdc(d::Tuple,i::Tuple) = rdc(rdc(Dict(),d),i)
    @everywhere rdc(d::Dict,i::Tuple) = begin d[i] = overlapped_code_ent(data_raw_code_dict, current_partition_pair, i, 2, n); d end
    @everywhere rdc(d::Dict,i::Dict) = merge!(d,i)
    @time begin
    results = @distributed (rdc) for part in parts
        part
    end
    end
end

@everywhere function overlapped_code_ent(dict, part_a, part_b, dict_index, ent_base)
    total_ent = 0.0
    for code_a in dict[part_a][dict_index]
        for code_b in dict[part_b][dict_index]
            # overlapped_freq = dot(code_a, code_b)
            # if overlapped_freq > 0 # base is always n in raw data
            #     total_ent += ent(overlapped_freq / ent_base)
            # end
            total_ent += ent(dot(code_a, code_b) / ent_base)
        end
    end
    return total_ent
end

@everywhere function encoding_cost(partition_code_dict, base=MI_BASE)
    encoded_data_cost = 0.0
    pattern_cost = 0.0
    freqs = Array{Int32, 1}(); # array of freqency in all partitions and all feature values
    for (part, code_dict) in partition_code_dict
        for (value, code_entry) in code_dict
            push!(freqs, code_entry[1])
        end
    end
    total_freq = sum(freqs)
    pattern_cost -= sum(log.(base, freqs / total_freq))
    encoded_data_cost -= sum(freqs .* log.(base, freqs / total_freq))
    singleton_freq_dict = Dict()
    for (part, code_dict) in partition_code_dict
        for value in keys(code_dict)
            merge!(+, singleton_freq_dict, StatsBase.countmap(flattenFeatureValues(value)))
        end
    end
    singleton_freqs = collect(values(singleton_freq_dict))
    total_singleton_freq = sum(singleton_freqs)
    singleton_cost = -sum(singleton_freqs .* log.(base, singleton_freqs / total_singleton_freq))
    return encoded_data_cost + singleton_cost + pattern_cost, singleton_freq_dict, total_freq, singleton_cost
end

function updated_init_part()
    two_by_two_range = collect(IterTools.subsets(names(data_raw), 2))
    two_by_two_ent_to_add = @distributed (rd_pair) for (partition_a, partition_b) in two_by_two_range
        ((partition_a, ), (partition_b, ))
    end
    partition_ig_pq = DataStructures.PriorityQueue() # (((F0), (F1)), ig) in order
    for (partition_pair, two_by_two_ent) in two_by_two_ent_to_add
        # information gain = entropy of first partition + entropy of second partition - entropy of overlapped part
        ig = (data_raw_code_dict[(partition_pair[1],)][1] + data_raw_code_dict[(partition_pair[2],)][1] - two_by_two_ent) / 2
        enqueue!(partition_ig_pq, ((partition_pair[1], ), (partition_pair[2], )), -ig) # min heap so take most negative
        println("pushed ", ((partition_pair[1], ), (partition_pair[2], )), ' ', -ig, " entry to priority queue for merging")
    end
end

function orig_init_part()
    for (partition_a, partition_b) in IterTools.subsets(names(data_raw), 2)
        sum_cost = 0.0
        for elem_a in data_raw_code_dict[(partition_a,)][2]
            for elem_b in data_raw_code_dict[(partition_b,)][2]
                # overlapped_freq = dot(elem_a, elem_b) # take & of two codes to get overlapped counts
                # if overlapped_freq > 0
                #     sum_cost += ent(overlapped_freq / n)
                # end
                sum_cost += ent(dot(elem_a, elem_b) / n)
            end
        end
        # information gain = entropy of first partition + entropy of second partition - entropy of overlapped part
        ig = (data_raw_code_dict[(partition_a,)][1] + data_raw_code_dict[(partition_b,)][1] - sum_cost) / 2
        enqueue!(partition_ig_pq, ((partition_a, ), (partition_b, )), -ig) # min heap so take most negative
        println("pushed ", ((partition_a, ), (partition_b, )), ' ', -ig, " entry to priority queue for merging")
    end
end

function orig_heavy_part()
    for (part, code_dict) in partition_code_dict # make parallel
        if part != current_partition_pair
            current_partition_pair_ent = data_raw_code_dict[current_partition_pair][1]
            part_ent = data_raw_code_dict[part][1]
            overlapped_ent = 0.0
            for code_a in data_raw_code_dict[current_partition_pair][2]
                for code_b in data_raw_code_dict[part][2]
                    # overlapped_freq = dot(code_a, code_b)
                    # if overlapped_freq > 0
                    #     overlapped_ent -= overlapped_freq / n * log(MI_BASE, overlapped_freq / n)
                    # end
                    overlapped_ent += ent(dot(code_a, code_b) / n)
                end
            end
            ig = (current_partition_pair_ent + part_ent - overlapped_ent) / (length(flattenPartitionPairs(part)) + length(flattenPartitionPairs(current_partition_pair)))
            enqueue!(partition_ig_pq, (current_partition_pair, part), -ig)
        end
    end
end

function take_more_pairs()
    partitions = flattenPartitionPairs(current_partition_pair)
    code_ranges = [1:length(data_raw_code_dict[p].codes) for p in partitions]
    sum_ent = 0.0
    data_raw_code_dict[current_partition_pair] = DataDictEntry(sum_ent, Vector())
    for code_ind in Iterators.product(code_ranges...)
        for dict_pair_ind in enumerate(code_ind)
            if dict_pair_ind[1] === 1
                global temp_code = data_raw_code_dict[partitions[1]].codes[dict_pair_ind[2]]
            else
                global temp_code .&= data_raw_code_dict[partitions[dict_pair_ind[1]]].codes[dict_pair_ind[2]]
            end
        end
        sum_ent += sum(temp_code)
        push!(data_raw_code_dict[current_partition_pair].codes, temp_code)
    end
    data_raw_code_dict[current_partition_pair] = DataDictEntry(sum_ent, data_raw_code_dict[current_partition_pair].codes)
end

function generate_data_basic()
    n = 10000000
    feature_value_dict = countmap(data_raw[!, :F150])
    data_raw_gen = sample(collect(keys(feature_value_dict)), Weights(values(feature_value_dict) ./ sum(values(feature_value_dict))), n)
    for (idx, field) in enumerate(names(data_raw))
        if 151 < idx
            println(field)
            feature_value_dict = countmap(data_raw[!, field])
            global data_raw_gen = hcat(data_raw_gen, sample(collect(keys(feature_value_dict)), Weights(values(feature_value_dict) ./ sum(values(feature_value_dict))), n))
        end
    end
end
