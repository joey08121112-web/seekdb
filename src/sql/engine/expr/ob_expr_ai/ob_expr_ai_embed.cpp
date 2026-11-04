/*
 * Copyright (c) 2025 OceanBase.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define USING_LOG_PREFIX SQL_ENG
#include "ob_expr_ai_embed.h"
#include "query/ai/ob_ai_endpoint_resolver.h"
#include "share/rc/ob_server_runtime.h"

using namespace oceanbase::common;
using namespace oceanbase::sql;

namespace oceanbase 
{
namespace sql 
{
ObExprAIEmbed::ObExprAIEmbed(common::ObIAllocator &alloc)
    : ObFuncExprOperator(alloc, 
                        T_FUN_SYS_AI_EMBED, 
                        N_AI_EMBED, 
                        MORE_THAN_ZERO,
                        NOT_VALID_FOR_GENERATED_COL, 
                        NOT_ROW_DIMENSION) 
{
}

ObExprAIEmbed::~ObExprAIEmbed() 
{
}

int ObExprAIEmbed::calc_result_typeN(ObExprResType &type,
                                     ObExprResType *types_stack,
                                     int64_t param_num,
                                     common::ObExprTypeCtx &type_ctx) const 
{
  UNUSED(type_ctx);
  UNUSED(types_stack);
  int ret = OB_SUCCESS;
  if (OB_UNLIKELY(param_num > 3 || param_num < 2)) {
    ObString func_name_(get_name());
    ret = OB_ERR_PARAM_SIZE;
    LOG_USER_ERROR(OB_ERR_PARAM_SIZE, func_name_.length(), func_name_.ptr());
  } else {
    types_stack[MODEL_IDX].set_calc_type(ObVarcharType);
    types_stack[MODEL_IDX].set_calc_collation_type(CS_TYPE_UTF8MB4_BIN);
    types_stack[CONTENT_IDX].set_calc_type(ObVarcharType);
    types_stack[CONTENT_IDX].set_calc_collation_type(CS_TYPE_UTF8MB4_BIN);
    if (param_num == 3) {
      if (ob_is_integer_type(types_stack[DIM_IDX].get_type())) {
        types_stack[DIM_IDX].set_calc_type(ObIntType);
        types_stack[DIM_IDX].set_precision(10);
        types_stack[DIM_IDX].set_scale(0);
      } else {
        ret = OB_INVALID_ARGUMENT;
        LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_embed, dimension parameter must be an integer, not a decimal or float");
      }
    }
    type.set_varchar();
    type.set_collation_type(CS_TYPE_UTF8MB4_BIN);
    type.set_collation_level(CS_LEVEL_COERCIBLE);
  }
  return ret;
}

// tail shared by the row-wise eval and the batch fallback: validates the
// optional dimension argument, resolves the model endpoint and requests one
// embedding. callers must have ruled out null model/content datums already.
static int eval_ai_embed_row(const ObExpr &expr, ObEvalCtx &ctx, ObDatum &res,
                             MultimodeAlloctor &temp_allocator, const ObString &model_id,
                             const ObString &content, const ObDatum *arg_dim)
{
  INIT_SUCC(ret);
  ObAIFuncExprInfo *info = nullptr;
  share::ObAiModelEndpointInfo resolved_endpoint;
  const share::ObAiModelEndpointInfo *endpoint_info = &resolved_endpoint;
  query::ObIAiEndpointResolver *endpoint_resolver =
      ::oceanbase::share::server_service<::oceanbase::query::ObIAiEndpointResolver>();
  if (model_id.empty() || content.empty()) {
    ret = OB_INVALID_ARGUMENT;
    LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_embed, model id or input is empty");
    res.set_null();
  }
  int64_t dim = 0;
  ObJsonInt *dim_json = nullptr;
  ObJsonObject *config = nullptr;
  if (OB_FAIL(ret)) {
  } else if (OB_NOT_NULL(arg_dim)) {
    if (arg_dim->is_null()) {
      // get_int() on a null datum is undefined; reject the NULL dimension
      // explicitly with the same wording as the non-positive case
      ret = OB_INVALID_ARGUMENT;
      LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_embed, dimension parameter must be a positive integer");
      res.set_null();
    } else if ((dim = arg_dim->get_int()) <= 0) {
      ret = OB_INVALID_ARGUMENT;
      LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_embed, dimension parameter must be a positive integer");
      res.set_null();
    } else if (OB_FAIL(ObAIFuncJsonUtils::get_json_object(temp_allocator, config))) {
    } else if (OB_FAIL(ObAIFuncJsonUtils::get_json_int(temp_allocator, dim, dim_json))) {
    } else if (OB_FAIL(config->add("dimensions", dim_json))) {
    }
  }
  if (OB_FAIL(ret)) {
  } else if (OB_FAIL(ObAIFuncUtils::get_ai_func_info(temp_allocator, model_id, info))) {
  } else if (OB_ISNULL(endpoint_resolver)) {
    ret = OB_ERR_UNEXPECTED;
  } else if (OB_FAIL(endpoint_resolver->resolve_by_model_name(
                 model_id, temp_allocator, resolved_endpoint))) {
  } else {
    ObAIFuncModel model(temp_allocator, *info, *endpoint_info);
    ObString result;
    if (OB_FAIL(model.call_dense_embedding(content, config, result))) {
    } else if (OB_FAIL(ObAIFuncUtils::set_string_result(expr, ctx, res, result))) {
    }
  }
  return ret;
}

int ObExprAIEmbed::eval_ai_embed(const ObExpr &expr, ObEvalCtx &ctx, ObDatum &res)
{
  INIT_SUCC(ret);
  ObDatum *arg_model_id = nullptr;
  ObDatum *arg_content = nullptr;
  ObDatum *arg_dim = nullptr;
  if (expr.arg_cnt_ == 3 ? OB_FAIL(expr.eval_param_value(ctx, arg_model_id, arg_content, arg_dim))
                         : OB_FAIL(expr.eval_param_value(ctx, arg_model_id, arg_content))) {
  } else if (arg_model_id->is_null() || arg_content->is_null()) {
    // null in, null out: standard null propagation. A NULL model id or
    // content row must yield a NULL result without failing the statement.
    res.set_null();
  } else {
    ObEvalCtx::TempAllocGuard tmp_alloc_g(ctx);
    MultimodeAlloctor temp_allocator(tmp_alloc_g.get_allocator());
    lib::ObMallocHookAttrGuard malloc_guard(lib::ObMemAttr(N_AI_EMBED));
    if (OB_FAIL(eval_ai_embed_row(expr, ctx, res, temp_allocator, arg_model_id->get_string(),
                                  arg_content->get_string(), arg_dim))) {
    }
  }
  return ret;
}

// cap on rows merged into one embeddings request: a wide vectorized batch is
// split into several bounded requests instead of one oversized POST
static constexpr int64_t MAX_EMBED_REQUEST_ROWS = 32;

int ObExprAIEmbed::eval_ai_embed_batch(const ObExpr &expr, ObEvalCtx &ctx,
                                       const ObBitVector &skip, const int64_t batch_size)
{
  int ret = OB_SUCCESS;
  ObDatumVector res_datum = expr.locate_expr_datumvector(ctx);
  ObBitVector &eval_flags = expr.get_evaluated_flags(ctx);
  const bool has_dim_arg = (expr.arg_cnt_ == 3);
  // evaluate every argument for the whole batch first; the row-wise eval
  // reaches the same state through eval_param_value
  for (int64_t i = 0; OB_SUCC(ret) && i < expr.arg_cnt_; i++) {
    if (OB_FAIL(expr.args_[i]->eval_batch(ctx, skip, batch_size))) {
    }
  }
  if (OB_SUCC(ret)) {
    ObDatumVector model_vec = expr.args_[MODEL_IDX]->locate_expr_datumvector(ctx);
    ObDatumVector content_vec = expr.args_[CONTENT_IDX]->locate_expr_datumvector(ctx);
    ObDatumVector dim_vec;
    if (has_dim_arg) {
      dim_vec = expr.args_[DIM_IDX]->locate_expr_datumvector(ctx);
    }
    ObEvalCtx::TempAllocGuard tmp_alloc_g(ctx);
    MultimodeAlloctor temp_allocator(tmp_alloc_g.get_allocator());
    lib::ObMallocHookAttrGuard malloc_guard(lib::ObMemAttr(N_AI_EMBED));
    // pass 1: classify every non-skipped row. NULL rows answer NULL on the
    // spot (null propagation, same as the row-wise eval); the rest is
    // collected for the remote call, with uniformity tracking for merging
    ObArray<int64_t> rows;
    ObArray<ObString> contents;
    ObString uniform_model;
    int64_t uniform_dim = 0;
    bool can_merge = true;
    for (int64_t j = 0; OB_SUCC(ret) && j < batch_size; j++) {
      if (skip.at(j) || eval_flags.at(j)) {
        continue;
      }
      eval_flags.set(j);
      const ObDatum *model_datum = model_vec.at(j);
      const ObDatum *content_datum = content_vec.at(j);
      int64_t row_dim = 0;
      if (model_datum->is_null() || content_datum->is_null()) {
        res_datum.at(j)->set_null();
        continue;
      }
      if (model_datum->get_string().empty() || content_datum->get_string().empty()) {
        ret = OB_INVALID_ARGUMENT;
        LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_embed, model id or input is empty");
        res_datum.at(j)->set_null();
        continue;
      }
      if (has_dim_arg) {
        const ObDatum *dim_datum = dim_vec.at(j);
        if (dim_datum->is_null() || (row_dim = dim_datum->get_int()) <= 0) {
          ret = OB_INVALID_ARGUMENT;
          LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_embed, dimension parameter must be a positive integer");
          res_datum.at(j)->set_null();
          continue;
        }
      }
      if (rows.empty()) {
        uniform_model = model_datum->get_string();
        uniform_dim = row_dim;
      } else if (model_datum->get_string() != uniform_model || row_dim != uniform_dim) {
        // heterogeneous model ids or dimensions cannot share one request;
        // fall back to per-row calls below instead of grouping
        can_merge = false;
      }
      if (OB_FAIL(rows.push_back(j))) {
      } else if (OB_FAIL(contents.push_back(content_datum->get_string()))) {
      }
    }
    if (OB_SUCC(ret)) {
      if (rows.empty()) {
        // nothing to request: every row was skipped, evaluated or NULL
      } else if (!can_merge || 1 == rows.count()) {
        // heterogeneous rows (or a lone one) keep the exact row-wise
        // semantics: one request per row through the shared tail
        for (int64_t k = 0; OB_SUCC(ret) && k < rows.count(); k++) {
          const int64_t j = rows[k];
          const ObDatum *arg_dim = has_dim_arg ? dim_vec.at(j) : nullptr;
          if (OB_FAIL(eval_ai_embed_row(expr, ctx, *res_datum.at(j), temp_allocator,
                                        model_vec.at(j)->get_string(),
                                        content_vec.at(j)->get_string(), arg_dim))) {
          }
        }
      } else {
        ObAIFuncExprInfo *info = nullptr;
        share::ObAiModelEndpointInfo resolved_endpoint;
        const share::ObAiModelEndpointInfo *endpoint_info = &resolved_endpoint;
        query::ObIAiEndpointResolver *endpoint_resolver =
            ::oceanbase::share::server_service<::oceanbase::query::ObIAiEndpointResolver>();
        ObJsonObject *config = nullptr;
        if (OB_ISNULL(endpoint_resolver)) {
          ret = OB_ERR_UNEXPECTED;
        } else if (OB_FAIL(ObAIFuncUtils::get_ai_func_info(temp_allocator, uniform_model, info))) {
        } else if (OB_FAIL(endpoint_resolver->resolve_by_model_name(
                       uniform_model, temp_allocator, resolved_endpoint))) {
        } else if (uniform_dim > 0) {
          ObJsonInt *dim_json = nullptr;
          if (OB_FAIL(ObAIFuncJsonUtils::get_json_object(temp_allocator, config))) {
          } else if (OB_FAIL(ObAIFuncJsonUtils::get_json_int(temp_allocator, uniform_dim, dim_json))) {
          } else if (OB_FAIL(config->add("dimensions", dim_json))) {
          }
        }
        if (OB_SUCC(ret)) {
          // one multi-input request per chunk: call_dense_embedding_vector_v2
          // validates the item count, per-item dimension and vector shape
          ObAIFuncModel model(temp_allocator, *info, *endpoint_info);
          for (int64_t start = 0; OB_SUCC(ret) && start < rows.count();
               start += MAX_EMBED_REQUEST_ROWS) {
            const int64_t chunk_cnt = MIN(MAX_EMBED_REQUEST_ROWS, rows.count() - start);
            ObArray<ObString> chunk_contents;
            ObArray<ObString> results;
            for (int64_t k = 0; OB_SUCC(ret) && k < chunk_cnt; k++) {
              if (OB_FAIL(chunk_contents.push_back(contents[start + k]))) {
              }
            }
            if (OB_FAIL(ret)) {
            } else if (OB_FAIL(model.call_dense_embedding_vector_v2(chunk_contents, config, results))) {
            } else if (results.count() != chunk_cnt) {
              ret = OB_ERR_UNEXPECTED;
            } else {
              for (int64_t k = 0; OB_SUCC(ret) && k < chunk_cnt; k++) {
                if (OB_FAIL(ObAIFuncUtils::set_string_result(
                        expr, ctx, *res_datum.at(rows[start + k]), results[k]))) {
                }
              }
            }
          }
        }
      }
    }
  }
  return ret;
}

int ObExprAIEmbed::cg_expr(ObExprCGCtx &expr_cg_ctx,
                           const ObRawExpr &raw_expr,
                           ObExpr &rt_expr) const 
{
  int ret = OB_SUCCESS;
  // TODO: support schema version match in plan cache for ai func
  // const ObRawExpr *model_key = raw_expr.get_param_expr(0);
  // if (OB_NOT_NULL(model_key)
  //     && (model_key->is_static_scalar_const_expr() || model_key->is_const_expr())
  //     && model_key->get_expr_type() != T_OP_GET_USER_VAR &&
  //     OB_NOT_NULL(expr_cg_ctx.schema_guard_)) {
  //   ObIAllocator *allocator = expr_cg_ctx.allocator_;
  //   ObExecContext *exec_ctx = expr_cg_ctx.session_->get_cur_exec_ctx();
  //   ObObj const_data;
  //   bool got_data = false;
  //   ObAIFuncExprInfo *info = nullptr;
  //   if (OB_ISNULL(allocator)) {
  //     ret = OB_ERR_UNEXPECTED;
  //     LOG_WARN("allocator is null", K(ret));
  //   } else if (OB_FAIL(ObSQLUtils::calc_const_or_calculable_expr(exec_ctx,
  //                                                         model_key,
  //                                                         const_data,
  //                                                         got_data,
  //                                                         *allocator))) {
  //     LOG_WARN("failed to calc offset expr", K(ret));
  //   } else if (!got_data || const_data.is_null()) {
  //   } else if (OB_FAIL(ObAIFuncUtils::get_ai_func_info(*allocator, const_data.get_string(), *expr_cg_ctx.schema_guard_, info))) {
  //     LOG_WARN("failed to get ai func info", K(ret), K(const_data.get_string()));
  //   } else {
  //     rt_expr.extra_info_ = info;
  //   }
  // }

  if (OB_SUCC(ret)) {
    rt_expr.eval_func_ = ObExprAIEmbed::eval_ai_embed;
    rt_expr.eval_batch_func_ = ObExprAIEmbed::eval_ai_embed_batch;
  }
  return ret;
}

} // namespace sql
} // namespace oceanbase
